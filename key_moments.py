import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx


def read_video(video_name, buffer_size):
    '''
    Read video and display basic information like number of frames,
    frame width, frame height and fps.
    Process 500 frames at once by calling process_frames_chunk()
    and passing a numpy array of 500 frames.
    :param video_name: name of the video.
    :param buffer_size: number of frames to be processed at once.
    :return: frameCount, fps
    '''
    cap = cv2.VideoCapture(video_name)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('Frame count: {}, Frame width: {}, Frame height: {}'.format\
              (frameCount, frameWidth, frameHeight))

    new_fc = frameCount - 500 # to avoid NAN during last chunk processing
    for i in range(0, new_fc, buffer_size):
        buffer = np.empty((buffer_size, frameHeight, frameWidth, 3), np.dtype('uint8'))

        fc = 0
        ret = True

        while fc < buffer_size and ret:
            ret, buffer[fc] = cap.read()
            fc += 1

        process_frames_chunk(buffer)
    cap.release()
    return frameCount, fps




def combine_clips(list_of_clips):
    pass


def process_frames_chunk(buffer):
    '''
    Detect yellow and orange color pixels
    Count yellow and orange pixels in each frame and append to fire_pixel_count list
    :param buffer: numpy array of frames
    :return:
    '''
    process_frames_chunk.counter += 1
    print(process_frames_chunk.counter, buffer.shape)
    for frame in buffer:
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(img_hsv, lower_bound, upper_bound)
        res = cv2.bitwise_and(frame, frame, mask=mask_yellow)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 70, 255, 0)
        fire_detected = cv2.countNonZero(thresh)
        fire_pixel_count.append(fire_detected)

def sorted_index_list():
    '''
    sort fire_pixel_count in descending order using np.argsort, reverse it
    :return: index of highest to lowest fire pixel counts
    '''
    s = np.array(fire_pixel_count)
    sort_index = np.argsort(s)
    clips = []

    sort_index_list = sort_index.tolist()
    sort_index_list.reverse()
    return sort_index_list

def key_moments_video(sort_index_list, file_name, duration, frame_count, fps_count):
    '''
    There is problem in logic. If i look at the frames with highest pixel counts,
    the frames are near each other which is resulting in repetetive key moments.
    So i took every 500th frame from sorted list but still the frames are near each other.
    :param sort_index_list:
    :param file_name: name of key_moments video file
    :param duration: duration in seconds
    :param frame_count: total frames
    :param fps_count: frames per second
    :return:
    '''
    clips = []
    count = 0
    for j in range(0, duration, 10):
        i = sort_index_list[count]
        count += 500
        print(i)
        if int(i / fps_count) - 5 < 1 or int(i / fps_count) + 5 > (frame_count / fps_count):
            continue # overcoming out of bounds but reduces duration by 10 sec
        clips.append(VideoFileClip('Assignment_firefighter1.mp4') \
                     .subclip(int(i / fps_count) - 5, int(i / fps_count) + 5).fx(vfx.fadein, 1).fx(vfx.fadeout, 1))
        # 5 seconds before and after target frame for context
    combined = concatenate_videoclips(clips)
    combined.write_videofile(file_name) # mp4 file creation


process_frames_chunk.counter = 0 # counting no of chunks the video is broken into
fire_pixel_count = [] # append count of fire pixel of each frame
lower_bound = np.array([20,50,50]) # lower bound of fire color
upper_bound = np.array([32,255,255]) # upper bound of fire color
print('successful imports')
frame_count, fps_count = read_video('Assignment_firefighter1.mp4', buffer_size=500) # reading video and processing in chunks
print('length of pixel_count: {}, max of pixel_count: {}'.\
                format(len(fire_pixel_count), max(fire_pixel_count)))
sorted_list = sorted_index_list() # sort fire pixels list
print('Sorted list: ', len(sorted_list))
key_moments_video(sorted_list, 'fire_fighter_3.mp4', 60, frame_count, fps_count) # generate key_moments video from fire pixels
print('successful run')
