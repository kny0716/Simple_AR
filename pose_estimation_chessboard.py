import numpy as np
import cv2 as cv


# The given video and calibration data
video_file = 'chessboard_video.mp4'
K = np.array([[1.83852409e+03 ,0.00000000e+00 ,1.00614444e+03],
 [0.00000000e+00 ,1.83928757e+03, 5.21005255e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeff = np.array([ 0.20100666 ,-1.00984075 ,-0.00586841,  0.00798829,  1.71240783])
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

points = []
sphere_center = np.array([4.5,3.5,-0.5]) * board_cellsize
sphere_radius = 0.5 * board_cellsize

for i in range(1,10):
    lat = np.pi * i / 10
    for j in range(20):
        lon = 2 * np.pi * j / 20
        x = sphere_center[0] + sphere_radius * np.sin(lat) * np.cos(lon)
        y = sphere_center[1] + sphere_radius * np.sin(lat) * np.sin(lon)
        z = sphere_center[2] + sphere_radius * np.cos(lat)
        points.append([x,y,z])
points = np.array(points, dtype=np.float32)


# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])


# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)
        sphere_2d , _= cv.projectPoints(points, rvec,tvec, K, dist_coeff)

        for i in range(8):  
            for j in range(19):  
                p1 = i * 20 + j
                p2 = (i + 1) * 20 + j
                p3 = i * 20 + (j + 1) % 20

                cv.line(img, tuple(np.int32(sphere_2d[p1][0])), tuple(np.int32(sphere_2d[p2][0])), (0, 255, 255), 1)
                cv.line(img, tuple(np.int32(sphere_2d[p1][0])), tuple(np.int32(sphere_2d[p3][0])), (0, 255, 255), 1)
                cv.line(img, tuple(np.int32(sphere_2d[p2][0])), tuple(np.int32(sphere_2d[p3][0])), (0, 255, 255), 1)

                p4 = (i + 1) * 20 + (j + 1) % 20
                cv.line(img, tuple(np.int32(sphere_2d[p2][0])), tuple(np.int32(sphere_2d[p4][0])), (0, 255, 255), 1)
                cv.line(img, tuple(np.int32(sphere_2d[p3][0])), tuple(np.int32(sphere_2d[p4][0])), (0, 255, 255), 1)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()