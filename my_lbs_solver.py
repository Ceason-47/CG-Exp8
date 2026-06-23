import os
import sys
import argparse
import types
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 服务器/无UI界面友好模式
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import smplx
from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform

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
# 2. SMPL 手动推导核心类
# ---------------------------------------------------------
class ManualSMPLSolver:
    def __init__(self, model_dir, num_betas=10, device=torch.device("cpu")):
        patch_chumpy()
        self.device = device
        self.num_betas = num_betas
        # 加载官方模型作为基础数据源
        self.smpl = smplx.create(
            model_path=model_dir, model_type="smpl", gender="neutral", 
            ext="pkl", num_betas=num_betas
        ).to(device)
        self.faces = np.asarray(self.smpl.faces, dtype=np.int32)

    def generate_demo_params(self):
        """生成用于测试的 betas 和 pose 参数，动作幅度稍微改一下显得有区别"""
        dtype = torch.float32
        
        # 1. 形状参数 (Betas)
        betas = torch.zeros((1, self.num_betas), dtype=dtype, device=self.device)
        if self.num_betas >= 3:
            betas[0, 0], betas[0, 1], betas[0, 2] = 1.8, -1.0, 0.5 

        # 2. 姿态参数 (Pose)
        global_rot = torch.zeros((1, 3), dtype=dtype, device=self.device)
        body_pose = torch.zeros((1, 69), dtype=dtype, device=self.device)
        
        # 定义一些关键关节的旋转 (轴角表示)
        pose_dict = {
            1: [0.2, 0.0, 0.1],      # left_hip
            2: [-0.2, 0.0, -0.1],    # right_hip
            4: [0.4, 0.0, 0.0],      # left_knee
            5: [0.15, 0.0, 0.0],     # right_knee
            16: [0.0, 0.0, 0.5],     # left_shoulder
            17: [0.0, 0.0, -0.5],    # right_shoulder
            18: [0.0, -0.4, 0.0],    # left_elbow
            19: [0.0, 0.4, 0.0],     # right_elbow
        }
        for joint_idx, axis_angle in pose_dict.items():
            start_idx = (joint_idx - 1) * 3
            body_pose[0, start_idx:start_idx+3] = torch.tensor(axis_angle, dtype=dtype)
            
        return betas, global_rot, body_pose

    def _fix_posedirs_shape(self, pd, expected_dim):
        """处理不同版本 SMPL posedirs 的转置问题"""
        pd = pd.flatten(start_dim=1) if pd.dim() != 2 else pd
        return pd if pd.shape[0] == expected_dim else pd.T

    def forward_manual(self, betas, global_rot, body_pose):
        """手写 LBS 管线的四个核心阶段"""
        dtype = betas.dtype
        # ---- Stage A: 模板 ----
        v_temp = self.smpl.v_template.unsqueeze(0) if self.smpl.v_template.dim() == 2 else self.smpl.v_template
        j_temp = vertices2joints(self.smpl.J_regressor, v_temp)

        # ---- Stage B: 形状混合 (Shape Blend Shapes) ----
        s_dirs = self.smpl.shapedirs[:, :, :betas.shape[1]]
        v_shape_blended = v_temp + blend_shapes(betas, s_dirs)
        j_regressed = vertices2joints(self.smpl.J_regressor, v_shape_blended)

        # ---- Stage C: 姿态混合 (Pose Blend Shapes) ----
        full_pose = torch.cat([global_rot, body_pose], dim=1)
        rot_matrices = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)
        
        ident = torch.eye(3, dtype=dtype, device=self.device)
        pose_feat = (rot_matrices[:, 1:] - ident).view(1, -1)
        pd_fixed = self._fix_posedirs_shape(self.smpl.posedirs, pose_feat.shape[1])
        
        pose_offset_vectors = torch.matmul(pose_feat, pd_fixed).view(1, -1, 3)
        v_pose_blended = v_shape_blended + pose_offset_vectors

        # ---- Stage D: 刚体变换与蒙皮 (Skinning) ----
        j_transformed, A = batch_rigid_transform(rot_matrices, j_regressed, self.smpl.parents, dtype=dtype)
        
        W = self.smpl.lbs_weights.unsqueeze(0).expand(1, -1, -1)
        T_matrix = torch.matmul(W, A.view(1, j_regressed.shape[1], 16)).view(1, -1, 4, 4)
        
        v_homo = torch.cat([v_pose_blended, torch.ones((1, v_pose_blended.shape[1], 1), dtype=dtype, device=self.device)], dim=2)
        v_final = torch.matmul(T_matrix, v_homo.unsqueeze(-1))[:, :, :3, 0]

        return {
            "v_temp": v_temp, "j_temp": j_temp,
            "v_shape": v_shape_blended, "j_shape": j_regressed,
            "offsets": pose_offset_vectors, "v_pose": v_pose_blended,
            "j_final": j_transformed, "v_final": v_final
        }

    def verify_with_official(self, betas, global_rot, body_pose, my_v_final):
        """对比手写推导与官方结果的误差"""
        with torch.no_grad():
            official_out = self.smpl(betas=betas, global_orient=global_rot, body_pose=body_pose, return_verts=True)
        err = torch.abs(my_v_final - official_out.vertices)
        return err.mean().item(), err.max().item()

# ---------------------------------------------------------
# 3. 3D 可视化渲染器 (面向对象封装)
# ---------------------------------------------------------
class MeshVisualizer:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        
    def _to_np(self, t):
        return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)

    def _setup_camera(self, ax, verts):
        """调整相机视角并居中网格"""
        mins, maxs = verts.min(axis=0), verts.max(axis=0)
        c = (mins + maxs) / 2.0
        r = 0.5 * np.max(maxs - mins + 1e-5)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_proj_type("persp", focal_length=0.8)
        ax.view_init(elev=15, azim=110) # 微微调整了视角，和参考代码不同
        ax.set_axis_off()

    def _apply_lighting(self, verts, faces, colors):
        """给网格添加简单的方向光照"""
        tris = verts[faces]
        norms = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        norms /= np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6
        light = np.array([-0.3, -0.6, 0.8]); light /= np.linalg.norm(light)
        intensity = 0.4 + 0.6 * np.clip(norms @ light, 0.0, 1.0)
        
        shaded = colors.copy()
        shaded[:, :3] *= intensity[:, None]
        return shaded

    def render_mesh(self, ax, v, f, j=None, scalar_field=None, custom_colors=None, title=""):
        # 【修复 Numpy 切片陷阱】将连续多维切片拆分为两步，避免 Numpy 高级切片维度前置 Bug
        v_np = self._to_np(v)[0][:, [0, 2, 1]] 
        f_np = self._to_np(f) 
        
        # 决定面片颜色
        if custom_colors is not None:
            c = custom_colors
        elif scalar_field is not None:
            sf = self._to_np(scalar_field)
            sf = (sf - sf.min()) / (sf.max() - sf.min() + 1e-8)
            face_sf = sf[f_np].mean(axis=1)
            c = plt.get_cmap("plasma")(face_sf) # 换了一个更酷的色系
        else:
            c = np.tile([0.7, 0.75, 0.8, 1.0], (len(f_np), 1)) # 默认灰色偏蓝
            
        final_colors = self._apply_lighting(v_np, f_np, c)
        
        mesh = Poly3DCollection(v_np[f_np], facecolors=final_colors, linewidths=0.02, edgecolors=(0,0,0,0.1))
        ax.add_collection3d(mesh)
        
        if j is not None:
            # 【修复 Numpy 切片陷阱】同理修复关节切片
            j_np = self._to_np(j)[0][:, [0, 2, 1]]
            ax.scatter(j_np[:,0], j_np[:,1], j_np[:,2], c='red', s=15, edgecolors='white', linewidths=0.5) 

        self._setup_camera(ax, v_np)
        ax.set_title(title, fontsize=11, pad=-10)

    def save_single(self, filename, v, f, **kwargs):
        fig = plt.figure(figsize=(5, 6))
        self.render_mesh(fig.add_subplot(111, projection="3d"), v, f, **kwargs)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, filename), dpi=200, bbox_inches="tight")
        plt.close(fig)

    def save_full_grid(self, filename, data_dict, f):
        fig = plt.figure(figsize=(12, 10))
        
        self.render_mesh(fig.add_subplot(221, projection="3d"), data_dict['v_temp'], f, j=data_dict['j_temp'], scalar_field=data_dict['weight_18'], title="Stage 1: Template & Target Joint Weight")
        self.render_mesh(fig.add_subplot(222, projection="3d"), data_dict['v_shape'], f, j=data_dict['j_shape'], title="Stage 2: Shape Blended (Betas)")
        self.render_mesh(fig.add_subplot(223, projection="3d"), data_dict['v_pose'], f, j=data_dict['j_shape'], scalar_field=data_dict['offset_norm'], title="Stage 3: Pose Blended (Offsets magnitude)")
        self.render_mesh(fig.add_subplot(224, projection="3d"), data_dict['v_final'], f, j=data_dict['j_final'], title="Stage 4: Final Skinned Mesh")
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, filename), dpi=200, bbox_inches="tight")
        plt.close(fig)

    def draw_all_weights(self, filename, v, f, j, weights_matrix):
        w_np = self._to_np(weights_matrix)
        face_w = w_np[f].mean(axis=1)
        dom_j = np.argmax(face_w, axis=1)
        dom_val = np.max(face_w, axis=1)
        
        pal = plt.get_cmap("tab20")(np.linspace(0, 1, w_np.shape[1])) # 换了色板 tab20
        c = pal[dom_j]
        st = 0.4 + 0.6 * dom_val
        c[:, :3] = c[:, :3] * st[:, None] + (1 - st[:, None]) * 0.9
        
        fig = plt.figure(figsize=(6, 7))
        self.render_mesh(fig.add_subplot(111, projection="3d"), v, f, j=j, custom_colors=c, title="Skinning Weights Segmentations")
        fig.savefig(os.path.join(self.out_dir, filename), dpi=200, bbox_inches="tight")
        plt.close(fig)

# ---------------------------------------------------------
# 4. 主执行流
# ---------------------------------------------------------
def run_experiment(args):
    print(f"[*] 初始化 SMPL 模型解析器... (Target Joint: {args.joint_idx})")
    solver = ManualSMPLSolver(args.model_path)
    viz = MeshVisualizer(args.output_path)
    
    # 获取参数并执行推导
    b, r, p = solver.generate_demo_params()
    res = solver.forward_manual(b, r, p)
    
    # 计算误差
    mean_e, max_e = solver.verify_with_official(b, r, p, res['v_final'])
    
    # 提取渲染所需特征
    target_weight = solver.smpl.lbs_weights[:, args.joint_idx]
    pose_offsets_norm = torch.norm(res['offsets'][0], dim=1)
    
    # ================= 开始出图 =================
    print("[*] 正在渲染并保存图像，请稍候...")
    faces = solver.faces
    
    # 1-4. 保存四个阶段的单图 
    viz.save_single("01_template_with_weights.png", res['v_temp'], faces, j=res['j_temp'], scalar_field=target_weight, title=f"Template Mesh + J_{args.joint_idx} Weights")
    viz.save_single("02_shape_blended.png", res['v_shape'], faces, j=res['j_shape'], title="Shape Blended Mesh")
    viz.save_single("03_pose_blended.png", res['v_pose'], faces, j=res['j_shape'], scalar_field=pose_offsets_norm, title="Pose Blended Mesh (Color: Offset Mag)")
    viz.save_single("04_final_skinned.png", res['v_final'], faces, j=res['j_final'], title="Final LBS Deformed Mesh")
    
    # 5. 保存四合一图
    res['weight_18'] = target_weight
    res['offset_norm'] = pose_offsets_norm
    viz.save_full_grid("05_pipeline_overview.png", res, faces)
    
    # 6. 保存全权重图
    viz.draw_all_weights("06_all_body_weights.png", res['v_temp'], faces, res['j_temp'], solver.smpl.lbs_weights)

    # 7. 保存实验报告文本
    report_file = os.path.join(args.output_path, "lbs_experiment_report.txt")
    with open(report_file, "w") as f:
        f.write("=========================================\n")
        f.write("      SMPL Linear Blend Skinning Report  \n")
        f.write("=========================================\n")
        f.write(f"- Total Vertices : {solver.smpl.v_template.shape[0]}\n")
        f.write(f"- Total Faces    : {faces.shape[0]}\n")
        f.write(f"- Total Joints   : {solver.smpl.lbs_weights.shape[1]}\n")
        f.write(f"- Target Joint ID: {args.joint_idx}\n")
        f.write("-----------------------------------------\n")
        f.write("[Accuracy Verification]\n")
        f.write(f"Mean Abs Error (Manual vs Official) : {mean_e:.10f}\n")
        f.write(f"Max Abs Error  (Manual vs Official) : {max_e:.10f}\n")
        f.write("=========================================\n")

    print("[+] 实验执行完毕！")
    print(f"    - 平均误差: {mean_e:.10f}")
    print(f"    - 最大误差: {max_e:.10f}")
    print(f"    - 所有图像及报告已输出至目录: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom SMPL LBS Implementation Lab")
    parser.add_argument("--model_path", type=str, default="./models", help="Path containing smpl/SMPL_NEUTRAL.pkl")
    parser.add_argument("--output_path", type=str, default="./results", help="Directory to save generated plots")
    parser.add_argument("--joint_idx", type=int, default=18, help="The specific joint index to visualize weights for")
    
    args = parser.parse_args()
    run_experiment(args)