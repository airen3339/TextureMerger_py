import os
import numpy as np
import math
import glob
import cv2
import struct

inputPath = 'input'
outputPath = 'output'
testPath = 'test'


class ImgMixer():
    def __init__(self):
        self.imgMap = []
        self.unit = 128

    def isDiff(self, diffVal):
        bName = self.getFileName(self.imgName)
        if 'FrontBG' in bName:
            return diffVal < 0.5
        else:
            return diffVal < 2

    def getExt(self, filePath):
        return filePath.split('.')[-1]

    def getFileName(self, filePath):
        base = os.path.basename(filePath)
        return os.path.splitext(base)[0]

    def getBinPath(self, filePath):
        return '%s.bin' % self.getFileName(filePath)

    def getExImgPath(self, filePath):
        return '%s/%s.jpg' % (outputPath, self.getFileName(filePath))

    def getMatPath(self):
        return '%s/mat.jpg' % outputPath

    def getFile(self, type, path=inputPath, filterFile=None):
        match = []
        filter = []
        if filterFile != None:
            with open(filterFile) as f:
                filter = f.readlines()
            filter = [a.strip() for a in filter]
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(type):
                    match.append(os.path.join(root, file))
        return [a for a in match if a not in filter]

    def makeMaps(self, inputImg):
        unitW, unitH = self.unit, self.unit

        h, w, _ = inputImg.shape

        xLength = w // unitW
        yLength = h // unitH

        indexMap = bytearray()
        header = struct.pack("2B", xLength, yLength)
        indexMap.extend(header)

        for y in range(yLength):
            for x in range(xLength):
                pimgX = x * unitW
                pimgY = y * unitH

                pimg = inputImg[pimgY:pimgY + unitH, pimgX:pimgX + unitW]
                resIndex = self.calUniqueImg(pimg)
                b = struct.pack('H', resIndex)
                indexMap.extend(b)

        return indexMap

    def getBrightness(self, img):
        grayImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        m = cv2.mean(grayImage)[:3][0]
        return m

    def getCmpSource(self, img):
        grayImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(
            grayImage, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return blackAndWhiteImage

    def imgCmp(self, img1, img2):
        imgGray1 = self.getCmpSource(img1)
        imgGray2 = self.getCmpSource(img2)
        res = cv2.absdiff(imgGray1, imgGray2)
        res = res.astype(np.uint8)
        percentage = (np.count_nonzero(res) * 100) / res.size
        return percentage

    def idConverter(self, id):
        isChange = False
        threshold = 0xfe
        if id // threshold != self.currentId:
            isChange = True
            self.currentId = id // threshold
        return isChange, id & threshold

    def calUniqueImg(self, partImg):
        isFound = False
        bImg = self.getBrightness(partImg)

        if len(self.imgMap) > 0:
            sortedSet = []
            for i, set in enumerate(self.imgMap):
                img = set[0]
                diff = self.imgCmp(img, partImg)
                sortedSet.append((i, diff, set))

            sortedSet = sorted(sortedSet, key=lambda item: item[1])
            for set in sortedSet:
                index, diff, img, b2 = set[0], set[1], set[2][0], set[2][1]
                if self.isDiff(diff):
                    if abs(b2 - bImg) < 2:
                        resIndex = index
                        isFound = True
                else:
                    break
                if isFound:
                    break
        if not isFound:
            set = partImg, bImg
            self.imgMap.append(set)
            resIndex = len(self.imgMap) - 1

        return resIndex

    def concatImg(self):
        unitW, unitH = self.unit, self.unit
        imgMapLen = len(self.imgMap)
        wCount = 20
        w, h = wCount * unitW, math.ceil(imgMapLen / wCount) * unitH
        dst = np.zeros((h, w, 3), np.uint8)
        for i, set in enumerate(self.imgMap):
            img = set[0]
            pimgX = i % wCount * unitW
            pimgY = i // wCount * unitH
            dst[pimgY:pimgY + unitH, pimgX:pimgX + unitW] = img
        return dst

    def openImg(self, imgName):
        self.imgName = imgName

        if '.png' in imgName:
            img = cv2.imread(imgName, cv2.IMREAD_UNCHANGED)
            trans_mask = img[:, :, 3] == 0
            img[trans_mask] = [0, 0, 255, 255]
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img = cv2.imread(imgName)

        return img

    def saveMat(self):
        concatImg = self.concatImg()
        self.saveImg(self.getMatPath(), concatImg)

    def saveImg(self, filePath, img):
        self.tryMakeDir(filePath)
        cv2.imwrite(filePath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

    def saveMaps(self, fName, indexMap):

        binPath = self.getBinPath(fName)
        outBinPath = '%s/%s' % (outputPath, binPath)

        self.tryMakeDir(outBinPath)
        with open(outBinPath, 'wb') as f:
            f.write(indexMap)

    def start(self, type=0):
        fileNames = self.getFile((".jpg", ".png"), inputPath)

        if type == 1:
            fileNames = [
                name for name in fileNames
                if os.path.basename(name).startswith('A')
                or os.path.basename(name).startswith('B')
            ]

        # corp and save unique images
        for fName in fileNames:
            img = self.openImg(fName)
            indexMap = self.makeMaps(img)
            self.saveMaps(fName, indexMap)
            print("complete: %s" % fName)
        self.saveMat()

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

    def getMaterial(self, matName):
        matImg = self.matting(matName)
        h, w, _ = matImg.shape
        unitW, unitH = self.unit, self.unit
        xLength = w // unitW
        yLength = h // unitH

        mats = []
        for y in range(yLength):
            for x in range(xLength):
                img = matImg[y * unitH:y * unitH + unitH,
                             x * unitW:x * unitW + unitW]
                mats.append(img)
        return mats

    def tryMakeDir(self, path):
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    def makeTestImg(self, filePath, mat):
        unitW, unitH = self.unit, self.unit
        with open(filePath, 'rb') as f:
            data = bytearray(f.read())
        xLength, yLength = struct.unpack("2B", data[:2])

        dst = np.zeros((yLength * unitH, xLength * unitW, 4), dtype=np.uint8)

        for i, (firstID, secondID) in enumerate(zip(data[2::2], data[3::2])):
            id = (firstID & 0xff) + (secondID << 8)
            img = mat[id]
            x = (i % xLength) * unitW
            y = (i // xLength * unitH)
            dst[y:y + unitH, x:x + unitW] = img

        output = '%s/%s.png' % (testPath, self.getFileName(filePath))
        self.tryMakeDir(output)
        cv2.imwrite(output, dst)

    def restore(self):
        binName = self.getFile(".bin", outputPath)
        matPath = self.getMatPath()
        mat = self.getMaterial(matPath)
        for bName in binName:
            self.makeTestImg(bName, mat)
