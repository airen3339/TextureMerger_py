import cv2
import os
import numpy as np

inputPath = 'input'
outputPath = 'output'
testPath = 'test'


class ImgReduce():
    def getFile(self, type, path=inputPath):
        match = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(type):
                    match.append(os.path.join(root, file))
        return match

    def rsplit(self, word, num=1):
        return word.rsplit('.', num)[0]

    def getTestImgPath(self, filePath):
        filePath = filePath.replace(".tran", "")
        return '%s/%s.png' % (testPath, self.rsplit(filePath))

    def getMatPath(self, filePath):
        if '.png' in filePath:
            return '%s/%s.tran.jpg' % (outputPath, self.rsplit(filePath))
        else:
            return '%s/%s.jpg' % (outputPath, self.rsplit(filePath))

    def openImg(self, imgName):
        if '.png' in imgName:
            img = cv2.imread(imgName, cv2.IMREAD_UNCHANGED)
            trans_mask = img[:, :, 3] == 0
            img[trans_mask] = [0, 0, 255, 255]
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img = cv2.imread(imgName)

        return img

    def tryMakeDir(self, path):
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    def saveImgs(self, filePath, img):
        outMatPath = self.getMatPath(filePath)
        self.tryMakeDir(outMatPath)
        cv2.imwrite(outMatPath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

    def matting(self, imgName):
        img = cv2.imread(imgName)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        mask = mask0 + mask1
        mask = cv2.bitwise_not(mask)

        img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

        res = cv2.bitwise_and(img_rgba, img_rgba, mask=mask)
        return res

    def makeTestImg(self, imgPath):
        if '.tran' in imgPath:
            matImg = self.matting(imgPath)
        else:
            matImg = cv2.imread(imgPath)
        resultPath = self.getTestImgPath(imgPath)
        self.tryMakeDir(resultPath)
        cv2.imwrite(resultPath, matImg)

    def start(self):
        fileNames = self.getFile((".jpg", ".png"))
        for fName in fileNames:
            img = self.openImg(fName)
            self.saveImgs(fName, img)

    def restore(self):
        imgNames = self.getFile((".jpg", ".png"), outputPath)
        for imgName in imgNames:
            self.makeTestImg(imgName)


if __name__ == "__main__":
    imgReduce = ImgReduce()
    # imgReduce.start()
    imgReduce.restore()
