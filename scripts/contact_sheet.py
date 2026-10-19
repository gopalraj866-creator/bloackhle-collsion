import cv2, glob, sys, numpy as np
fs = sorted(glob.glob(sys.argv[1] + "/f_*.png")); ims = [cv2.resize(cv2.imread(f), (640, 360)) for f in fs]
while len(ims) % 4: ims.append(np.zeros_like(ims[0]))
cv2.imwrite(sys.argv[2], np.vstack([np.hstack(ims[i:i + 4]) for i in range(0, len(ims), 4)]))
