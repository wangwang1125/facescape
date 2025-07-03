import trimesh, numpy as np
from scipy.spatial.distance import cdist

# make trimesh from vertices and faces
def make_trimesh(verts, faces, vert_colors = None):
    if vert_colors is None:
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    else:
        return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vert_colors, process=False)

# to replace trimesh.load
def load_ori_mesh(fn):
    return trimesh.load(fn, 
                        resolver = None, 
                        split_object = False, 
                        group_material = False, 
                        skip_materials = False, 
                        maintain_order = True, 
                        process = False)

# mesh_A and mesh_B are trimesh class
# warn: if require_array == True, the output - errors_B2A, errors_A2B  may contains NaN
def compute_chamfer(mesh_A, mesh_B, require_array = False):
    # B to A: find closest points on mesh_A for each vertex in mesh_B
    closests_B2A, _, _ = mesh_A.nearest.on_surface(mesh_B.vertices)
    errors_B2A = np.linalg.norm(mesh_B.vertices - closests_B2A, axis=1)
    
    # A to B: find closest points on mesh_B for each vertex in mesh_A  
    closests_A2B, _, _ = mesh_B.nearest.on_surface(mesh_A.vertices)
    errors_A2B = np.linalg.norm(mesh_A.vertices - closests_A2B, axis=1)
    
    errors_all = np.concatenate((errors_A2B, errors_B2A))
    chamfer_dist = np.mean(errors_all[~np.isnan(errors_all)])
    if require_array is False:
        return chamfer_dist
    else:
        return chamfer_dist, errors_B2A, errors_A2B
    
# depth==bg_val means backgroud, will not be visualized
def depth2mesh(depth, threshold=15, bg_val=0.0):
    h, w = depth.shape
    indices_map = np.indices((h, w))
    
    # make vertices
    verts = np.concatenate((indices_map.transpose((1, 2, 0)), 
                           np.expand_dims(depth, axis = 2)), axis = 2).reshape((h * w, 3))
    
    # generate valid mask according to difference of depth and bg_val
    depth_diff = np.zeros((h, w, 3))
    depth_diff[:,:,0] = depth - np.roll(depth, -1, axis=0)
    depth_diff[:,:,1] = depth - np.roll(depth, -1, axis=1)
    depth_diff[:,:,2] = depth - np.roll(np.roll(depth, -1, axis=0), -1, axis=1)
    depth_diff_max = np.max(np.abs(depth_diff), axis=2)
    valid_mask = (depth_diff_max<threshold) * (depth!=bg_val)
    
    # make faces
    face_array = np.zeros((h, w, 6), dtype = int)
    
    face_array[:,:,0] = indices_map[0] * w + indices_map[1]
    face_array[:,:,1] = face_array[:,:,0] + 1
    face_array[:,:,2] = (indices_map[0]+1) * w + indices_map[1]
    face_array[:,:,3] = face_array[:,:,2] + 1
    face_array[:,:,4] = face_array[:,:,2]
    face_array[:,:,5] = face_array[:,:,0] + 1
    
    # mask out and reshape
    valid_mask_c6 = np.stack((valid_mask, )*6, axis=2)    
    faces = face_array[valid_mask_c6]
    faces = faces.reshape((len(faces)//3, 3))
    
    # make trimesh
    mesh = make_trimesh(verts, faces)
    mesh.remove_unreferenced_vertices()

    return mesh

# cyl is the displacement map on cyl coordinate
# cyl==bg_val means backgroud, will not be visualized
def cylinder2mesh(cyl, threshold=15, bg_val=0.0, cyl_len=150):
    h, w = cyl.shape
    indices_map = np.indices((h, w))
    
    # make verts
    phi_arr = indices_map[1] * 2 * np.pi / 255
    verts_arr = np.stack((cyl * (-np.sin(phi_arr)), 
                          np.stack((-np.linspace(-cyl_len, cyl_len, w), )*h, axis=1), 
                          cyl * (-np.cos(phi_arr))), axis = 2)    
    verts = verts_arr.reshape((h*w, 3))
    
    # generate valid mask according to difference of depth and bg_val
    cyl_diff = np.zeros((h, w, 3))
    cyl_diff[:,:,0] = cyl - np.roll(cyl, -1, axis=0)
    cyl_diff[:,:,1] = cyl - np.roll(cyl, -1, axis=1)
    cyl_diff[:,:,2] = cyl - np.roll(np.roll(cyl, -1, axis=0), -1, axis=1)
    cyl_diff_max = np.max(np.abs(cyl_diff), axis=2)    
    valid_mask = (cyl_diff_max<threshold) * (cyl!=bg_val)
    
    # make faces
    face_array = np.zeros((h, w, 6), dtype = int)
    
    face_array[:,:,0] = indices_map[0] * w + indices_map[1]
    face_array[:,:,1] = face_array[:,:,0] + 1
    face_array[:,:,2] = (indices_map[0]+1) * w + indices_map[1]
    face_array[:,:,3] = face_array[:,:,2] + 1
    face_array[:,:,4] = face_array[:,:,2]
    face_array[:,:,5] = face_array[:,:,0] + 1
    
    # mask out and reshape
    valid_mask_c6 = np.stack((valid_mask, )*6, axis=2)
    faces = face_array[valid_mask_c6]
    faces = faces.reshape((len(faces)//3, 3))
    
    # make trimesh
    mesh = make_trimesh(verts, faces)
    mesh.remove_unreferenced_vertices()
    return mesh


# # depth==bg_val means backgroud, will not be visualized
# def depth2mesh_slow(depth, threshold=15, bg_val=0.0):
#     h = depth.shape[0]
#     w = depth.shape[1]
#     verts, faces = [], []
#     # make verts list
#     for i in range(h):
#         for j in range(w):
#             verts.append([float(i), float(j), float(depth[i, j])])
    
#     # make face list
#     for i in range(h - 10):
#         for j in range(w - 10):
#             if i < 10 or j < 10:
#                 continue
#             if depth[i, j] == bg_val:
#                 continue
#             localpatch = np.copy(depth[i - 1:i + 2, j - 1:j + 2])
#             dy_u = localpatch[0, :] - localpatch[1, :]
#             dx_l = localpatch[:, 0] - localpatch[:, 1]
#             dy_d = localpatch[0, :] - localpatch[-1, :]
#             dx_r = localpatch[:, 0] - localpatch[:, -1]
#             dy_u = np.abs(dy_u)
#             dx_l = np.abs(dx_l)
#             dy_d = np.abs(dy_d)
#             dx_r = np.abs(dx_r)
#             if np.max(dy_u) < threshold and \
#                     np.max(dx_l) < threshold and \
#                     np.max(dy_d) < threshold and \
#                     np.max(dx_r) < threshold:
#                 faces.append([int(i * w + j), int(i * w + j + 1),
#                               int((i + 1) * w + j)])
#                 faces.append([int((i + 1) * w + j + 1), int((i + 1) * w + j),
#                               int(i * w + j + 1)])
    
#     # make trimesh
#     mesh = make_trimesh(verts, faces)
#     mesh.remove_unreferenced_vertices()

#     return mesh

# def cylinder2mesh_slow(npy, threshold=15, bg_val=0.0):
#     h = npy.shape[0]
#     w = npy.shape[1]
#     verts, faces = [], []
#     y_list = np.linspace(-150, 150, 256)

#     # make verts list
#     for i in range(h):
#         for j in range(w):
#             phi = j * 2 * np.pi / 255
#             x = npy[i, j] * (-np.sin(phi))
#             y = y_list[255 - i]
#             z = npy[i, j] * (-np.cos(phi))
#             verts.append([x, y, z])

#     # make face list
#     for i in range(h - 3):
#         for j in range(w - 3):
#             if i < 3 or j < 3:
#                 continue
#             if npy[i, j] == bg_val:
#                 continue
#             localpatch = np.copy(npy[i - 1:i + 2, j - 1:j + 2])
#             dy_u = localpatch[0, :] - localpatch[1, :]
#             dx_l = localpatch[:, 0] - localpatch[:, 1]
#             dy_d = localpatch[0, :] - localpatch[-1, :]
#             dx_r = localpatch[:, 0] - localpatch[:, -1]
#             dy_u = np.abs(dy_u)
#             dx_l = np.abs(dx_l)
#             dy_d = np.abs(dy_d)
#             dx_r = np.abs(dx_r)
#             if np.max(dy_u) < threshold and \
#                     np.max(dx_l) < threshold and \
#                     np.max(dy_d) < threshold and \
#                     np.max(dx_r) < threshold:
#                 faces.append([int(i * w + j), int(i * w + j + 1),
#                               int((i + 1) * w + j)])
#                 faces.append([int((i + 1) * w + j + 1), int((i + 1) * w + j),
#                               int(i * w + j + 1)])

#     # make trimesh
#     mesh = make_trimesh(verts, faces)
#     mesh.remove_unreferenced_vertices()
#     return mesh

def procrustes_analysis(source_points, target_points):
    """
    基于Procrustes分析进行点集配准
    """
    # 中心化
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    
    # 计算最优旋转矩阵
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 确保旋转矩阵的行列式为正
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 计算缩放因子
    scale = np.sum(S) / np.sum(source_centered**2)
    
    # 计算平移向量
    t = target_centroid - scale * R @ source_centroid
    
    return R, t, scale

def apply_icp_alignment(pred_mesh, gt_mesh, pred_keypoints_idx, real_keypoints_idx, max_iterations=20, tolerance=1e-6):
    """
    使用关键点进行初始配准，然后进行ICP优化
    """
    # 提取关键点
    pred_keypoints = pred_mesh.vertices[pred_keypoints_idx]
    gt_keypoints = gt_mesh.vertices[real_keypoints_idx]
    
    print(f"使用 {len(pred_keypoints)} 个关键点进行ICP配准")
    
    # 基于关键点的初始配准（Procrustes分析）
    R, t, scale = procrustes_analysis(pred_keypoints, gt_keypoints)
    
    # 应用初始变换到预测mesh
    aligned_mesh = pred_mesh.copy()
    aligned_mesh.vertices = scale * (aligned_mesh.vertices @ R.T) + t
    
    print(f"初始配准完成，缩放因子: {scale:.4f}")
    
    # ICP迭代优化
    prev_error = float('inf')
    for iteration in range(max_iterations):
        # 找到最近点对应
        closest_points, _, _ = gt_mesh.nearest.on_surface(aligned_mesh.vertices)
        
        # 计算当前误差
        distances = np.linalg.norm(aligned_mesh.vertices - closest_points, axis=1)
        current_error = np.mean(distances)
        
        print(f"ICP迭代 {iteration+1}: 平均误差 = {current_error:.6f}")
        
        # 检查收敛
        if abs(prev_error - current_error) < tolerance:
            print(f"ICP在第 {iteration+1} 次迭代后收敛")
            break
        
        # 更新变换
        try:
            R_iter, t_iter, scale_iter = procrustes_analysis(aligned_mesh.vertices, closest_points)
            aligned_mesh.vertices = scale_iter * (aligned_mesh.vertices @ R_iter.T) + t_iter
        except:
            print(f"ICP在第 {iteration+1} 次迭代时失败，停止优化")
            break
        
        prev_error = current_error
    
    return aligned_mesh

def icp_alignment_ubsd(pred_mesh, gt_mesh):
    """
    专门为UBSD方法设计的ICP配准函数
    """
    # UBSD模型的关键点索引
    pred_keypoints_idx = np.array([28125, 20228, 19546, 19595, 20328, 40793, 40812])
    real_keypoints_idx = np.array([3021, 8899, 7666, 5937, 6610, 6712, 7940])
    
    # 检查索引是否有效
    if np.max(pred_keypoints_idx) >= len(pred_mesh.vertices):
        print(f"警告: UBSD预测mesh顶点数({len(pred_mesh.vertices)})小于最大关键点索引({np.max(pred_keypoints_idx)})")
        return pred_mesh
    
    if np.max(real_keypoints_idx) >= len(gt_mesh.vertices):
        print(f"警告: GT mesh顶点数({len(gt_mesh.vertices)})小于最大关键点索引({np.max(real_keypoints_idx)})")
        return pred_mesh
    
    try:
        aligned_mesh = apply_icp_alignment(pred_mesh, gt_mesh, pred_keypoints_idx, real_keypoints_idx)
        return aligned_mesh
    except Exception as e:
        print(f"ICP配准失败: {e}")
        return pred_mesh