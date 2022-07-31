import os
import sys
import subprocess
import argparse
import shutil

try:
    from face_landmark_detection import makeCorrespondence
    from delaunay import makeDelaunay
    from faceMorph import makeMorphs
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from face_morphing.morphing.face_landmark_detection import makeCorrespondence
    from face_morphing.morphing.delaunay import makeDelaunay
    from face_morphing.morphing.faceMorph import makeMorphs


def do_morphing(predictor, img1, img2, dur_frames, result_dir):
    [size, img1, img2, list1, list2, list3] = makeCorrespondence(predictor, img1, img2)
    if size[0] == 0:
        print('Sorry, but I couldn\'t find a face in the image ' + size[1])
        return
    list4 = makeDelaunay(size[1], size[0], list3)
    makeMorphs(dur_frames, img1, img2, list1, list2, list4, size, result_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img1', help='The First Image')
    parser.add_argument('img2', help='The Second Image')
    parser.add_argument('dur_frames', type=int, help='The Duration in frames')
    parser.add_argument('res', help='The Resultant Video')
    args = parser.parse_args()

    with open(args.img1, 'rb') as image1, open(args.img2, 'rb') as image2:
        do_morphing('shape_predictor_68_face_landmarks.dat', image1, image2, args.dur_frames, args.res)
