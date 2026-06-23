import os
import sys
import types
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 后台渲染模式
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import smplx
from smplx.lbs import vertices2joints, batch_rodrigues, batch_rigid_transform, blend_shapes
import imageio

# ---------------------------------------------------------
# 1. 兼容性补丁 (处理老版本 SMPL 的 chumpy 依赖问题)
# ---------------------------------------------------------
class _FakeChumpy:
    def __setstate__(self, state):
        self.__dict__.update(state)
    def _array(self):
        return self.r if hasattr(self, "r") else self.x
    def __array__(self, dtype=None):
        return np.asarray(self._array(), dtype=dtype)
    @property
    def shape(self): return np.asarray(self).shape
    def __len__(self): return len(np.asarray(self))
    def __getitem__(self, item): return np.asarray(self)[item]

def patch_chumpy():
    if "chumpy.ch" not in sys.modules:
        m_chumpy = types.ModuleType("chumpy")
        m_ch_sub = types.ModuleType("chumpy.ch")
        _FakeChumpy.__name__ = "Ch"
        _FakeChumpy.__module__ = "chumpy.ch"
        m_ch_sub.Ch = _FakeChumpy
        m_chumpy.ch = m_ch_sub
        sys.modules["chumpy"] = m_chumpy
        sys.modules["chumpy.ch"] = m_ch_sub

# ---------------------------------------------------------
# 2. SMPL 动画生成器核心类
# ---------------------------------------------------------
class SMPLAnimator:
    def __init__(self, model_dir, device=torch.device("cpu")):
        patch_chumpy()
        self.device = device
        # 加载官方模型
        self.smpl = smplx.create(
            model_path=model_dir, model_type="smpl", gender="neutral", 
            ext="pkl", num_betas=10
        ).to(device)
        self.faces = np.asarray(self.smpl.faces, dtype=np.int32)
        
        # 固定 Shape 参数 (保持体型不变)
        self.fixed_betas = torch.zeros((1, 10), dtype=torch.float32, device=self.device)
        self.fixed_betas[0, 0] = 1.5  # 稍微调整一下体型

    def compute_mesh_at_pose(self, body_pose):
        """给定姿态参数，快速计算最终的蒙皮网格坐标 (调用官方前向传播以保证速度和稳定)"""
        global_rot = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = self.smpl(
                betas=self.fixed_betas,
                global_orient=global_rot,
                body_pose=body_pose,
                return_verts=True
            )
        return output.vertices[0].detach().cpu().numpy(), output.joints[0].detach().cpu().numpy()

# ---------------------------------------------------------
# 3. 3D 渲染与 GIF 导出模块
# ---------------------------------------------------------
class AnimationVisualizer:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # 预先计算好固定的相机范围，防止动画抖动
        self.fixed_center = np.array([0.0, 0.0, 0.0])
        self.fixed_radius = 1.2

    def _setup_camera(self, ax):
        """使用固定的包围盒限制，确保每一帧的相机视野绝对静止"""
        c = self.fixed_center
        r = self.fixed_radius
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_proj_type("persp", focal_length=0.8)
        # 调整一个能看清手臂弯曲和侧面的好视角
        ax.view_init(elev=15, azim=70)
        ax.set_axis_off()

    def _apply_lighting(self, verts, faces, colors):
        tris = verts[faces]
        norms = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        norms /= np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6
        light = np.array([-0.3, -0.6, 0.8]); light /= np.linalg.norm(light)
        intensity = 0.4 + 0.6 * np.clip(norms @ light, 0.0, 1.0)
        shaded = colors.copy()
        shaded[:, :3] *= intensity[:, None]
        return shaded

    def render_and_save_frame(self, frame_idx, v, f, weight_scalar, title):
        # 坐标系转换 (Y/Z互换适应 Matplotlib)
        v_np = v[:, [0, 2, 1]]
        f_np = f
        
        # 根据权重计算面片颜色 (高权重区域变红，低权重区域偏灰白)
        sf = weight_scalar
        sf = (sf - sf.min()) / (sf.max() - sf.min() + 1e-8)
        face_sf = sf[f_np].mean(axis=1)
        colors = plt.get_cmap("coolwarm")(face_sf)
        
        final_colors = self._apply_lighting(v_np, f_np, colors)
        
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(111, projection="3d")
        
        mesh = Poly3DCollection(v_np[f_np], facecolors=final_colors, linewidths=0.0, edgecolors='none')
        ax.add_collection3d(mesh)
        
        self._setup_camera(ax)
        ax.set_title(title, fontsize=14, pad=0)
        
        frame_path = os.path.join(self.frames_dir, f"frame_{frame_idx:03d}.png")
        fig.tight_layout()
        fig.savefig(frame_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return frame_path

    def compile_gif(self, frame_paths, gif_name="pose_animation.gif", fps=15):
        print(f"[*] 正在将 {len(frame_paths)} 帧图片合成为 GIF...")
        images = []
        for path in frame_paths:
            images.append(imageio.imread(path))
        
        out_path = os.path.join(self.out_dir, gif_name)
        imageio.mimsave(out_path, images, fps=fps)
        print(f"[+] 动图已保存至: {out_path}")

# ---------------------------------------------------------
# 4. 主执行流
# ---------------------------------------------------------
def run_animation_lab():
    model_path = "./models"
    output_path = "./results"
    
    # 我们选择关节 18 (左手手肘 Left Elbow) 进行弯曲动画
    target_joint = 18
    # 动画参数
    num_frames = 30
    max_angle = 2.0  # 约 114 度，明显的手臂弯曲
    
    print(f"[*] 初始化动画引擎... (目标关节: {target_joint}, 总帧数: {num_frames})")
    animator = SMPLAnimator(model_path)
    viz = AnimationVisualizer(output_path)
    
    # 提取目标关节的权重，用于渲染时染色
    target_weight = animator.smpl.lbs_weights[:, target_joint].detach().cpu().numpy()
    
    frame_paths = []
    
    # 生成每一帧
    for i in range(num_frames):
        # 使用正弦函数让动画平滑地来回摆动 (0 -> max_angle -> 0)
        progress = np.sin((i / (num_frames - 1)) * np.pi)
        current_angle = progress * max_angle
        
        # 更新 Pose 参数
        body_pose = torch.zeros((1, 69), dtype=torch.float32)
        start_idx = (target_joint - 1) * 3
        # 让手肘绕着特定轴系旋转（这里是弯曲手肘的经典轴）
        body_pose[0, start_idx:start_idx+3] = torch.tensor([0.0, -current_angle, 0.0])
        
        # 计算当前帧的网格
        v_current, _ = animator.compute_mesh_at_pose(body_pose)
        
        # 渲染并保存图片
        title = f"Joint {target_joint} Animation\nAngle: {current_angle:.2f} rad"
        path = viz.render_and_save_frame(i, v_current, animator.faces, target_weight, title)
        frame_paths.append(path)
        
        # 打印进度条
        print(f"\r[> ] 渲染进度: {i+1}/{num_frames} 帧", end="")
    
    print("\n[*] 所有关键帧渲染完毕！")
    
    # 将图片编译为 GIF
    viz.compile_gif(frame_paths, "joint_18_bend.gif", fps=15)
    print("\n[+] 选做实验完美完成！")

if __name__ == "__main__":
    run_animation_lab()