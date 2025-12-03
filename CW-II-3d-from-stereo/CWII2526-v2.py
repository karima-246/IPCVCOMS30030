'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Coursework Part 2
3-D simulator

Andrew Calway andrew@cs.bris.ac.uk
'''

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse


'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication
    
    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix
    
    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n,m = points.shape
    if m == 4:
        new_points = points
    else:
        new_points = np.concatenate([points, np.ones((n,1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3,:]
    new_points = new_points[:3,:].transpose()
    return new_points

def check_dup_locations(y, z, loc_list):
    for (loc_y, loc_z) in loc_list:
        if loc_y == y and loc_z == z:
            return True


# print("here", flush=True)
if __name__ == '__main__': 

    ####################################
    ### Take command line arguments ####
    ####################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', dest='num', type=int, default=6, 
                        help='number of spheres')    
    parser.add_argument('--sph_rad_min', dest='sph_rad_min', type=int, default=10, 
                        help='min sphere  radius x10')
    parser.add_argument('--sph_rad_max', dest='sph_rad_max', type=int, default=16, 
                        help='max sphere  radius x10')
    parser.add_argument('--sph_sep_min', dest='sph_sep_min', type=int, default=4, 
                       help='min sphere  separation')
    parser.add_argument('--sph_sep_max', dest='sph_sep_max', type=int, default=8, 
                       help='max sphere  separation')
    parser.add_argument('--display_centre', dest='bCentre', action='store_true',
                        help='open up another visualiser to visualise centres')
    parser.add_argument('--coords', dest='bCoords', action='store_true')

    args = parser.parse_args()

    if args.num<=0:
        print('invalidnumber of spheres')
        exit()

    if args.sph_rad_min>=args.sph_rad_max or args.sph_rad_min<=0:
        print('invalid max and min sphere radii')
        exit()
    	
    if args.sph_sep_min>=args.sph_sep_max or args.sph_sep_min<=0:
        print('invalid max and min sphere separation')
        exit()
	
    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    box_mesh=o3d.geometry.TriangleMesh.create_box(width=h,height=0.05,depth=w)
    box_H=np.array(
                 [[1, 0, 0, -h/2],
                  [0, 1, 0, -0.05],
                  [0, 0, 1, -w/2],
                  [0, 0, 0, 1]]
                )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ['plane']
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    prev_loc = []
    GT_cents, GT_rads = [], []
    for i in range(args.num):
        # add sphere name
        name_list.append(f'sphere_{i}')

        # create sphere with random radius
        size = random.randrange(args.sph_rad_min, args.sph_rad_max, 2)/10
        sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_list.append(sph_mesh)
        RGB_list.append([0., 0.5, 0.5])

        # create random sphere location
        step = random.randrange(int(args.sph_sep_min),int(args.sph_sep_max),1)
        x = random.randrange(int(-h/2+2), int(h/2-2), step)
        z = random.randrange(int(-w/2+2), int(w/2-2), step)
        while check_dup_locations(x, z, prev_loc):
            x = random.randrange(int(-h/2+2), int(h/2-2), step)
            z = random.randrange(int(-w/2+2), int(w/2-2), step)
        prev_loc.append((x, z))

        GT_cents.append(np.array([x, size, z, 1.]))
        GT_rads.append(size)
        sph_H = np.array(
                    [[1, 0, 0, x],
                     [0, 1, 0, size],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]
                )
        H_list.append(sph_H)

    # arrange plane and sphere in the space
    obj_meshes = []
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    # add optional coordinate system
    if args.bCoords:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        obj_meshes = obj_meshes+[coord_frame]
        RGB_list.append([1., 1., 1.])
        name_list.append('coords')


    ###################################
    #### Setup camera orientations ####
    ###################################

    # set camera pose (world to camera)
    # # camera init 
    # # placed at the world origin, and looking at z-positive direction, 
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)      
    # print(H_init)

    # camera_0 (world to camera)
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    H0_wc = np.array(
                [[1,            0,              0,  0],
                [0, np.cos(theta), -np.sin(theta),  0], 
                [0, np.sin(theta),  np.cos(theta), 20], 
                [0, 0, 0, 1]]
            )

    # camera_1 (world to camera)
    theta = np.pi * (80+random.uniform(-10, 10))/180.
    H01 = np.array(
                [[np.cos(theta),  0, np.sin(theta), 0],
                 [0,              1, 0,             0],
                 [-np.sin(theta), 0, np.cos(theta), 0],
                 [0, 0, 0, 1]]
            )
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    H1_1 = np.array(
                [[1, 0,            0,              0],
                [0, np.cos(theta), -np.sin(theta), -4],
                [0, np.sin(theta), np.cos(theta),  20],
                [0, 0, 0, 1]]
            )
    H1_wc = np.matmul(H1_1, H01)
    render_list = [(H0_wc, 'view0.png', 'depth0.png'), 
                   (H1_wc, 'view1.png', 'depth1.png')]

#####################################################
    # NOTE: This section relates to rendering scenes in Open3D, details are not
    # critical to understanding the lab, but feel free to read Open3D docs
    # to understand how it works.
    
    # set up camera intrinsic matrix needed for rendering in Open3D
    img_width=640
    img_height=480
    f=415 # focal length
    # image centre in pixel coordinates
    ox=img_width/2-0.5 
    oy=img_height/2-0.5
    K = o3d.camera.PinholeCameraIntrinsic(img_width,img_height,f,f,ox,oy)

    # Rendering RGB-D frames given camera poses
    # create visualiser and get rendered views
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = K
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_width, height=img_height, left=0, top=0)
    for m in obj_meshes:
        vis.add_geometry(m)
    ctr = vis.get_view_control()
    for (H_wc, name, dname) in render_list:
        cam.extrinsic = H_wc
        ctr.convert_from_pinhole_camera_parameters(cam,True)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(name, True)
        vis.capture_depth_image(dname, True)
    vis.run()
    vis.destroy_window()
##################################################

    # load in the images for post processings
    img0 = cv2.imread('view0.png', -1)
    dep0 = cv2.imread('depth0.png', -1)
    img1 = cv2.imread('view1.png', -1)
    dep1 = cv2.imread('depth1.png', -1)

    # visualise sphere centres
    pcd_GTcents = o3d.geometry.PointCloud()
    pcd_GTcents.points = o3d.utility.Vector3dVector(np.array(GT_cents)[:, :3])
    pcd_GTcents.paint_uniform_color([1., 0., 0.])
    if args.bCentre:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, left=0, top=0)
        for m in [obj_meshes[0], pcd_GTcents]:
            vis.add_geometry(m)

    
    ###################################
    '''
    In lectures, the essential matrix for finding epipolar lines in the right image is defined to be E = RS, where:
    R = rotation from camera R to camera L
    S = skew-symmetric matrix, i.e. translation from camera L to camera R

    Following this, I have defined my essential matrix for finding epipolar lines in VC1 as E = R^TS = SR, where:
    R = rotation from camera 0 to camera 1
    S = cross product of A with translation vector from camera 0 to camera 1
    '''

    '''
    Task 3: Circle detection
    Hint: use cv2.HoughCircles() for circle detection.
    https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    Write your code here
    '''
    ###################################
    
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    gray0_blurred = cv2.medianBlur(gray0, 5)
    gray1_blurred = cv2.medianBlur(gray1, 5)

    centers0 = []
    centers1 = []

    circles0 = cv2.HoughCircles(gray0_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=40, param1=82, param2=41, minRadius=0, maxRadius=0)
    circles1 = cv2.HoughCircles(gray1_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=40, param1=82, param2=41, minRadius=0, maxRadius=0)

    cent_rads0 = {}
    radii0 = []
    circles0 = np.uint16(np.around(circles0))
    for i in circles0[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        # Draw the circle center
        cv2.circle(img0, center, 2, (0, 255, 0), 3)
        # Draw the circle outline
        cv2.circle(img0, center, radius, (255, 0, 0), 3)

        centerh = [i[0], i[1], 1]
        centers0.append(centerh)

        cent_rads0[center] = radius
        radii0.append(radius)
    print(cent_rads0)

    cent_rads1 = {}
    radii1 = []
    circles1 = np.uint16(np.around(circles1))
    for i in circles1[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        # Draw the circle center
        cv2.circle(img1, center, 2, (0, 255, 0), 3)
        # Draw the circle outline
        cv2.circle(img1, center, radius, (255, 0, 0), 3)

        centerh = [i[0], i[1], 1]
        centers1.append(centerh)

        cent_rads1[center] = radius
        radii1.append(radius)

    cv2.imwrite("view0.png", img0)
    cv2.imwrite("view1.png", img1)


    ###################################
    '''
    Task 4: Epipolar line
    Hint: Compute Essential & Fundamental Matrix
          Draw lines with cv2.line() function
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
    
    Write your code here
    '''
    ###################################
    # Transformation matrix from camera 0 -> camera 1
    H01 = np.matmul(H1_wc, np.linalg.inv(H0_wc))
    # Extract rotation
    R = H01[:3, :3]
    T01 = H01[:3, 3]

    # compute essential matrix
    S = np.array([[0, -T01[2], T01[1]],
                [T01[2], 0, -T01[0]],
                [-T01[1], T01[0], 0]
        ])
    E = S @ R

    # compute fundamental matrix
    F = np.transpose(np.linalg.inv(K.intrinsic_matrix)) @ E @ np.linalg.inv(K.intrinsic_matrix)

    for center0 in centers0:
        # Compute epipolar line for each circle center
        epipolar_line = F @ center0
        a, b, c = epipolar_line

        # Draw the line and center on the image
        height, width = img1.shape[:2]
        # Compute two points on the line (at image boundaries)
        if abs(b) > abs(a):  # Line is more horizontal
            x1 = 0
            y1 = int((-c - a * x1) / b)
            x2 = width - 1
            y2 = int((-c - a * x2) / b)
        else:  # Line is more vertical
            y1 = 0
            x1 = int((-c - b * y1) / a)
            y2 = height - 1
            x2 = int((-c - b * y2) / a)

        cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.rectangle(img1,(center0),(center0),(0,0,255),3)
    cv2.imwrite("epipolar_view1.png", img1)


    ###################################
    '''
    Task 5: Find correspondences

    Write your code here
    '''
    ###################################
    # Test each centre in C1 to see which is the closest inlier, i.e. which makes p'^T1Fp'0 closest to =0
    correspondences = {}
    for i, p0 in enumerate(centers0):
        errors = []
        for p1 in centers1:
            # Ensure points are column vectors shape (3,1)

            # Compute the epipolar constraint p1^T F p0
            error = float(np.transpose(p1) @ F @ p0)
            errors.append(error)

        # find the center1 with the smallest absolute constraint violation
        best_match_index = np.argmin(np.abs(errors))
        correspondences[i] = best_match_index

    for i in range(len(centers0)):
        print(f"center {i} MATCH center {correspondences[i]}")

    # print(correspondences)

        
            

    ###################################
    '''
    Task 6: 3-D locations of sphere centres

    Write your code here
    '''
    ###################################
    H10 = np.matmul(H0_wc, np.linalg.inv(H1_wc))
    # Extract rotation
    R10 = H10[:3, :3]
    T10 = H10[:3, 3]

    est_cents = []
    est_centsh = []

    for c0, c1 in correspondences.items():
        p0 = np.linalg.inv(K.intrinsic_matrix) @ np.array(centers0[c0]).reshape(3,)
        p1 = np.linalg.inv(K.intrinsic_matrix) @ np.array(centers1[c1]).reshape(3,)

        p0 = p0 / np.linalg.norm(p0)
        p1 = p1 / np.linalg.norm(p1)

        p1_0 = R10 @ p1

        H = np.vstack([
            p0,
            - p1_0,
            - np.cross(p0, p1_0)
        ])
        a, b, c = np.array(np.linalg.inv(H) @ T10)

        p = a * p0 
        p_hom = np.hstack([p, 1])

        P_world = np.linalg.inv(H0_wc) @ p_hom
        P_world = P_world[:3] 

        est_cents.append(P_world)
        est_centsh.append(np.hstack([P_world,1]))


        
        

    ###################################
    '''
    Task 7: Evaluate and Display the centres

    Write your code here
    '''
    ###################################

    pcd_MYcents = o3d.geometry.PointCloud()
    pcd_MYcents.points = o3d.utility.Vector3dVector(np.array(est_centsh)[:, :3])
    pcd_MYcents.paint_uniform_color([0., 1., 0.])
    if args.bCentre:
        for m in [pcd_MYcents]:
            vis.add_geometry(m)

    errors = []
    for est in est_centsh:
        # Compute distances from this estimate to all ground-truth centres
        diffs = GT_cents - est 
        dists = np.linalg.norm(diffs, axis=1)  # Euclidean distance
        min_dist = np.min(dists)  # closest ground-truth
        errors.append(min_dist)

    errors = np.array(errors)
    print("Errors:", errors) 

    vis.run()
    vis.destroy_window()

    ###################################
    '''
    Task 8: 3-D radius of spheres

    Write your code here
    '''
    ###################################
    # Detect circles in both images → gives their 2D circle centres and 2D circle radii.

    est_rads = []
    for i, (cent, rad) in enumerate(cent_rads0.items()):
        i1 = correspondences[i]
        items = list(cent_rads1.items())
        cent1, rad1 = items[i1]

        # Get the center in image coordinates (pixels)
        center_2d = np.array([cent[0], cent[1], 1.0])
        
        # Get an edge point in image coordinates (pixels)
        edge_2d = np.array([cent[0] + rad, cent[1], 1.0])
        
        # Unproject both center and edge to 3D rays from camera
        center_ray = np.linalg.inv(K.intrinsic_matrix) @ center_2d
        edge_ray = np.linalg.inv(K.intrinsic_matrix) @ edge_2d
        
        # Normalize the rays
        center_ray = center_ray / np.linalg.norm(center_ray)
        edge_ray = edge_ray / np.linalg.norm(edge_ray)
        
        # Get the estimated center in world coordinates
        est_center_world = est_cents[i]
        
        # Transform world center back to camera 0 coordinates
        est_center_cam = H0_wc[:3, :3] @ est_center_world + H0_wc[:3, 3]
        
        # Depth of the center along its ray
        depth = np.dot(est_center_cam, center_ray)
        
        # 3D position of edge point at same depth
        edge_3d_cam = depth * edge_ray
        
        # Compute radius as distance between center and edge in 3D
        radius_3d = np.linalg.norm(edge_3d_cam - est_center_cam)
        
        est_rads.append(radius_3d)

    print(f"3D Radii: {est_rads}")

    # Convert to numpy array
    est_rads = np.array(est_rads)
        
    # est_rads = []
    # for i, (cent, rad) in enumerate(cent_rads0.items()):
    #     i1 = correspondences[i]
    #     items = list(cent_rads1.items())
    #     cent1, rad1 = items[i1]

    #     p_edge = np.array([cent[0], cent[1]])
    #     p_edge[0] = p_edge[0] + rad
    #     P_edge = np.array([p_edge[0]*z/f, p_edge[1]*z/f, z])

    #     p1_edge = np.array([cent1[0], cent1[1]])
    #     p1_edge[0] += rad1
    #     P1_edge = np.array([p1_edge[0]*z/f, p1_edge[1]*z/f, z])

    #     est = est_cents[i]

    #     # how do i find the 3D radius? euclidean distance, or just x component distance?
    #     Rad0 = np.linalg.norm(P_edge - est)
    #     Rad1 = np.linalg.norm(P1_edge - est)
    #     Rad = (Rad0 + Rad1) / 2
    #     est_rads.append(Rad)

    # print(f"3D Radii: {est_rads}")

    ###################################
    '''
    Task 9: Display the spheres

    Write your code here:
    '''
    ###################################

    def transform_points(points, H):
        """Transform points by homogeneous transformation matrix H"""
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed = (H @ points_h.T).T
        return transformed[:, :3]

    # Assuming you have:
    # - GT_cents: list of original sphere centers (from your original code)
    # - GT_rads: list of original sphere radii
    # - est_cents: your estimated centers
    # - est_rads: your estimated radii

    # Recreate the plane
    h, w = 24, 12
    box_mesh = o3d.geometry.TriangleMesh.create_box(width=h, height=0.05, depth=w)
    box_H = np.array([[1, 0, 0, -h/2],
                    [0, 1, 0, -0.05],
                    [0, 0, 1, -w/2],
                    [0, 0, 0, 1]])
    box_mesh.vertices = o3d.utility.Vector3dVector(
        transform_points(np.asarray(box_mesh.vertices), box_H)
    )
    box_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    box_mesh.compute_vertex_normals()

    all_meshes = [box_mesh]

    # Recreate original spheres (ground truth) in cyan
    for cent, rad in zip(GT_cents, GT_rads):
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=rad)
        sph_H = np.array([[1, 0, 0, cent[0]],
                        [0, 1, 0, cent[1]],
                        [0, 0, 1, cent[2]],
                        [0, 0, 0, 1]])
        sph.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(sph.vertices), sph_H)
        )
        sph.paint_uniform_color([0., 0.5, 0.5])  # Cyan - original spheres
        sph.compute_vertex_normals()
        all_meshes.append(sph)

    # Create your estimated spheres in red/orange
    # Convert est_cents and est_rads to proper format
    est_cents = np.array(est_cents)
    est_rads = np.array(est_rads)

    # Print for debugging - remove this later
    print(f"Est_cents shape: {est_cents.shape}")
    print(f"Est_rads shape: {est_rads.shape}")
    print(f"Sample est_cent: {est_cents[0] if len(est_cents) > 0 else 'empty'}")
    print(f"Sample est_rad: {est_rads[0] if len(est_rads) > 0 else 'empty'}")

    for i in range(len(est_cents)):
        cent = est_cents[i]
        rad = est_rads[i]
        
        # Handle different possible formats
        if hasattr(cent, 'shape') and len(cent.shape) > 0:
            if cent.shape[0] >= 3:
                x, y, z = float(cent[0]), float(cent[1]), float(cent[2])
            else:
                continue
        else:
            x, y, z = float(cent[0]), float(cent[1]), float(cent[2])
        
        # Ensure radius is a scalar
        rad = float(rad) if hasattr(rad, '__iter__') and not isinstance(rad, str) else float(rad)
        
        # Create sphere mesh
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=rad)
        sph_H = np.array([[1, 0, 0, x],
                        [0, 1, 0, y],
                        [0, 0, 1, z],
                        [0, 0, 0, 1]])
        sph.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(sph.vertices), sph_H)
        )
        sph.paint_uniform_color([1., 0.3, 0.])  # Orange/red - estimated spheres
        sph.compute_vertex_normals()
        all_meshes.append(sph)

    # Visualize everything
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)

    for mesh in all_meshes:
        vis.add_geometry(mesh)

    vis.run()
    vis.destroy_window()


    ###################################
    '''
    Task 10: Investigate impact of noise added to relative pose

    Write your code here:
    '''
    ###################################
