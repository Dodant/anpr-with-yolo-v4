import matplotlib.pyplot as plt
import pytesseract
import numpy as np
import argparse
import imutils
import math
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to the images to be classified")
args = vars(ap.parse_args())

img = cv2.imread(args["images"])
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original (Gray)', img_gray)
cv2.waitKey(0)

img_blur = cv2.bilateralFilter(img_gray, 10, 50, 50)
cv2.imshow('Blur', img_blur)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_dilation = cv2.dilate(img_blur, kernel, iterations = 1)
cv2.imshow('Dilation', img_dilation)
cv2.waitKey(0)

# img_threshold = cv2.adaptiveThreshold(img_dilation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
# cv2.imshow('Threshold', img_threshold)
# cv2.waitKey(0)

_, img_threshold = cv2.threshold(img_dilation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('Threshold', img_threshold)
cv2.waitKey(0)

img_canny = cv2.Canny(img_threshold, 50, 150, apertureSize=3)
cv2.imshow('Canny', img_canny)
cv2.waitKey(0)

lines = cv2.HoughLines(img_canny, 1, np.pi/180, 150)

angleSum = 0
img_ori = img_threshold.copy()

for i in range(len(lines)):
    for rho, theta in lines[i]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho
        x1, y1 = int(x0 + 1000*(-b)),int(y0 + 1000*(a))
        x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))

        cv2.line(img_ori, (x1,y1), (x2,y2), (0,0,255), 3)
        angleSum += math.degrees(math.atan2(y2 - y1, x2 - x1))

angle = angleSum/len(lines)

res = np.vstack((img_ori.copy(), img_ori))
cv2.imshow('img', img_ori)
cv2.waitKey(0)

rows, cols = img.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
img_rotate = cv2.warpAffine(img_threshold.copy(), M, (cols, rows))
cv2.imshow('Rotation', img_rotate)
cv2.waitKey(0)

contours, _ = cv2.findContours(img_rotate, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(img_rotate, contours, -1, (0,255,0), 2)
cv2.imshow('Contours', image)
cv2.waitKey(0)

ori_img = img_rotate.copy()
contours,_ = cv2.findContours(ori_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_dict = []
pos_cnt = list()
box1 = list()

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(ori_img, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)

    # insert to dict
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })

cv2.imshow('Contour Candidate', ori_img)
cv2.waitKey(0)

# ---------------------------------------------------------

ori_img = img_rotate.copy()
count = 0

for d in contours_dict:
    rect_area = d['w']*d['h'] # 영역 크기
    aspect_ratio = d['w'] / d['h']

    if 0.3 <= aspect_ratio <= 1.0 and 100 <= rect_area:
        cv2.rectangle(ori_img,(d['x'],d['y']),(d['x']+d['w'],d['y']+d['h']),(0,255,0),2)
        d['idx'] = count
        count += 1
        pos_cnt.append(d)

cv2.imshow('Possible Candidate', ori_img)
cv2.waitKey(0)

# ----------------------------------------------

MAX_DIAG_MULTIPLYER = 5 # contourArea의 대각선 x5 안에 다음 contour가 있어야함
MAX_ANGLE_DIFF = 12.0  # contour와 contour 중심을 기준으로 한 각도가 n 이내여야함
MAX_AREA_DIFF = 0.5 # contour간에 면적 차이가 크면 인정하지 않겠다.
MAX_WIDTH_DIFF = 0.8 # contour간에 너비 차이가 크면 인정 x
MAX_HEIGHT_DIFF = 0.2 # contour간에 높이 차이가 크면 인정 x
MIN_N_MATCHED = 3 # 위의 조건을 따르는 contour가 최소 3개 이상이어야 번호판으로 인정
ori_img = img_rotate.copy()

def find_number(contour_list):
    matched_result_idx = []

    # contour_list[n]의 keys = dict_keys(['contour', 'x', 'y', 'w', 'h', 'cx', 'cy', 'idx'])
    for d1 in contour_list:
        matched_contour_idx = []
        for d2 in contour_list:      # for문을 2번 돌면서 contour끼리 비교해줄 것
            if d1['idx'] == d2['idx']:   # idx가 같다면 아예 동일한 contour이기에 패스
                continue

            dx = abs(d1['cx']-d2['cx'])  # d1, d2 중앙점 기준으로 x축의 거리
            dy = abs(d1['cy']-d2['cy'])  # d1, d2 중앙점 기준으로 y축의 거리
            # 이를 구한 이유는 대각 길이를 구하기 위함 / 피타고라스 정리

            # 기준 Contour 사각형의 대각선 길이 구하기
            diag_len = np.sqrt(d1['w']**2+d1['w']**2)

            # contour 중심간의 대각 거리
            distance = np.linalg.norm(np.array([d1['cx'],d1['cy']]) - np.array([d2['cx'],d2['cy']]))

            # 각도 구하기
            # 빗변을 구할 때, dx와 dy를 알기에 tan세타 = dy / dx 로 구할 수 있다.
            # 여기서 역함수를 사용하면    세타 =  arctan dy/dx 가 된다.
            if dx == 0:
                angle_diff = 90   # x축의 차이가 없다는 것은 다른 contour가 위/아래에 위치한다는 것
            else:
                angle_diff = np.degrees(np.arctan(dy/dx))  # 라디안 값을 도로 바꾼다.

            # 면적의 비율 (기준 contour 대비)
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w']*d1['h'])
            # 너비의 비율
            width_diff = abs(d1['w']-d2['w']) / d1['w']
            # 높이의 비율
            height_diff = abs(d1['h']-d2['h']) / d2['h']

            # 이제 조건에 맞는 idx만을 matched_contours_idx에 append할 것이다.
            if distance < diag_len * MAX_DIAG_MULTIPLYER and angle_diff < MAX_ANGLE_DIFF \
            and area_diff < MAX_AREA_DIFF and width_diff < MAX_WIDTH_DIFF \
            and height_diff < MAX_HEIGHT_DIFF:
                # 계속 d2를 번갈아 가며 비교했기에 지금 d2 넣어주고
                matched_contour_idx.append(d2['idx'])

        # d1은 기준이었으니 이제 append
        matched_contour_idx.append(d1['idx'])

        # 앞서 정한 후보군의 갯수보다 적으면 탈락
        if len(matched_contour_idx) < MIN_N_MATCHED:
            continue

        # 최종 contour를 입력
        matched_result_idx.append(matched_contour_idx)

        # 최종에 들지 못한 아닌애들도 한 번 더 비교
        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contour_idx:
                unmatched_contour_idx.append(d4['idx'])

        # np.take(a,idx)   a배열에서 idx를 뽑아냄
        unmatched_contour = np.take(pos_cnt,unmatched_contour_idx)

        # 재귀적으로 한 번 더 돌림
        recursive_contour_list = find_number(unmatched_contour)

        # 최종 리스트에 추가
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx

result_idx = find_number(pos_cnt)

matched_result = []

for idx_list in result_idx:
    matched_result.append(np.take(pos_cnt,idx_list))

# pos_cnt 시각화

for r in matched_result:
    for d in r:
        cv2.rectangle(ori_img,(d['x'],d['y']),(d['x']+d['w'],d['y']+d['h']),(0,255,0),2)

cv2.imshow('Possible Candidate', ori_img)
cv2.waitKey(0)

# --------------------------------------------

PLATE_WIDTH_PADDING = 1.3
PLATE_HEIGHT_PADDING = 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []
ori_img = img_rotate.copy()

for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

    # 합집합 구하는 것 처럼 교집합([0]['x']) 제거
    # 그리고 패딩
    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    # 평균 구하고 패딩
    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

    # 원하는 부분만 잘라냄
    img_cropped = cv2.getRectSubPix(
        ori_img,
        patchSize=(int(plate_width), int(plate_height)),
        center=(int(plate_cx), int(plate_cy))
    )
    # h/w < Min   or   Max < h/w < Min  해당하면 패스  해당하지 않을경우 append
    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or \
    img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        continue

    plate_imgs.append(img_cropped)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })

cv2.imshow('Possible Candidate', img_cropped)
cv2.waitKey(0)

print(pytesseract.image_to_string(img_cropped,
    lang='cuskor',
    config='-c tessedit_char_whitelist=0123456789가나다라마바사아자거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호배 --psm 7 --oem 3'))

cv2.destroyAllWindows()



# sobelx = cv2.convertScaleAbs(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3))
# sobely = cv2.convertScaleAbs(cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3))
# sobel = cv2.addWeighted(sobelx, 1, sobely, 1, 0)
# cv2.imshow('Laplacian', sobel)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# erosion = cv2.erode(image, kernel, iterations = 1)
# dilation = cv2.dilate(image, kernel, iterations = 1)
#
# opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE,kernel)
# gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
# tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
# blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
#
# images = [image, erosion, opening, image, dilation, closing, gradient, tophat, blackhat]
# titles = ['Original','Erosion','Opening','Original','Dilation','Closing', 'Gradient', 'Tophat','Blackhot']
#
# print(imagePath)
#
# for i in range(9):
#     print(titles[i], pytesseract.image_to_string(images[i], lang='kor'))
