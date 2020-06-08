
def detectPlates(self):

        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # squareKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # rectKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
        regions = []

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

        add = cv2.add(gray, tophat)
        subtract = cv2.subtract(add, blackhat)

        blackhat = cv2.GaussianBlur(blackhat, (5, 5), 0)
        blackhat2 = cv2.GaussianBlur(subtract, (5, 5), 0)

        thresh1 = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh2 = cv2.threshold(blackhat2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh3 = cv2.adaptiveThreshold(blackhat, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV,
                                        19,
                                        9)

        thresh3 = cv2.dilate(thresh3, None, iterations=2)

        thresh = cv2.adaptiveThreshold(blackhat2, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       19,
                                       20)

        thresh = cv2.subtract(thresh2, thresh)
        thresh = cv2.bitwise_and(thresh, thresh3)

        cv2.imshow("result", thresh)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            (w, h) = cv2.boundingRect(c)[2:]
            aspectRatio = w / float(h)

            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.cv.BoxPoints(rect)) if imutils.is_cv2() else cv2.boxPoints(rect)

            # if (aspectRatio > 2 and aspectRatio < 6) and h > self.minPlateH and w > self.minPlateW:
            if 1000 < w * h and 2 < aspectRatio < 5):
                regions.append(box)

        return regions
