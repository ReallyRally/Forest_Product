import cv2
import numpy as np
import glob

def load_image(img_path):
    all_images = []
    for name in ('*.png', '*.jpg', '.bmp', '.tiff'):
        all_images.extend(glob.glob(img_path + name))
    #print(all_images)
    print(f"이미지 전체 개수 : {len(all_images)}")

    for idx, f in enumerate(all_images):
        fname = f.split('\\')[-1]
        fname = fname.split('.')[0]
        print(fname)
        img = cv2.imread(f)
        cv2.imshow('test', img)

        YCbCr = luminance_separated2(img, fname)
        D = threshold_Y(YCbCr, fname)
        Is1 = morphologic(D, fname)
        inverted_Segmentation(img, YCbCr, Is1, fname)
        if cv2.waitKey(0) & 0xFF == 27:
            break


def luminance_separated2(img, idx):
    '''
    Step - 0
    RGB 이미지를 YCbCr 색공간으로 변환합니다.
    '''
    # Y, Cb, Cr 분할 매트릭스------------------(논문 참조)
    transform_matrix = np.array([[65.481, 128.553, 24.966],
                                 [-37.797, -74.203, 112.0],
                                 [112.0, -93.787, -18.214]])
    offset = np.array([[16], [128], [128]])

    # 이미지의 RGB를 0~1로 정규화 합니다
    img_normalized = img.astype('float32') / 255

    # 변환 행렬을 이용하여 RGB 이미지를 YCbCr 색공간으로 변환합니다.
    # 이는 전체 이미지에 대한 행렬 연산으로 수행되며, 이는 픽셀 별 연산보다 훨씬 빠릅니다.
    YCbCr = np.tensordot(img_normalized, transform_matrix.T, axes=([2], [0])) + offset.T

    # 결과 이미지를 저장합니다.
    cv2.imwrite(f'00.Results/00.Y_Cb_Cr/{idx}_Y_CB_CR.png', YCbCr.astype('uint8'))

    return YCbCr.astype('uint8')

def luminance_separated(img, idx):
    '''
    Step - 0
    '''
    # Y, Cb Cr 분할 매트릭스------------------(논문 참조)
    transform_matrix = np.array([[65.481, 128.553, 24.966],
                                [-37.797, -74.203, 112.0],
                                 [112.0, -93.787, -18.214]])
    # offset 설정
    offset = np.array([16, 128, 128])

    # 이미지의 RGB를 0~1로 정규화 합니다
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_normalized = img.astype('float32')/ 255

    R, G, B = cv2.split(img_normalized)

    YCbCr = np.empty_like(img_normalized)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            rgb_pixel = np.array([R[y, x], G[y, x], B[y, x]])
            YCbCr[y, x] = np.dot(transform_matrix, rgb_pixel) + offset

    Y, Cb, Cr = cv2.split(YCbCr)

    #YCbCr_255 = np.clip(YCbCr, 0, 1) * 255
    cv2.imwrite(f'00.Results/00.Y_Cb_Cr/{idx}_Y_CB_CR.png', YCbCr)
    return YCbCr.astype('uint8')

def threshold_Y(YCbCr, idx):
    #YCbCr에서 Y 성분을 가져옵니다.
    Y_ch = YCbCr[:, :, 0]
    threshold = 170
    _, D = cv2.threshold(Y_ch, threshold, 255, cv2.THRESH_BINARY)
    #D = np.where(Y_ch > threshold, 0, 1)
    cv2.imwrite(f'00.Results/00.Y_Cb_Cr/{idx}_Y_CB_CR_00thr.png', D)

    return D

def morphologic(S1, idx):
    #S1 = (S1).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    # 오프닝 연산 (노이즈 제거)
    opening = cv2.morphologyEx(S1, cv2.MORPH_OPEN, kernel)
    # 클로징 (내부 공백 제거)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    D = S1

    Is1 = cv2.bitwise_xor(D, opening)
    Is2 = cv2.bitwise_xor(Is1, closing)

    cv2.imwrite(f'00.Results/00.Y_Cb_Cr/{idx}_Y_CB_CR_01Is1.png', Is1)
    cv2.imwrite(f'00.Results/00.Y_Cb_Cr/{idx}_Y_CB_CR_02Is2.png', Is2)

    return Is2

def inverted_Segmentation(orginal,YCbCr, Is1, idx):
    Y, Cb, Cr = cv2.split(YCbCr)

    # (a) 과정: YCbCr 채널과 Is1을 AND 연산하여 Ib를 생성합니다.
    Iby = cv2.bitwise_and(Y, Is1)
    Ibcb = cv2.bitwise_and(Cb, Is1)
    Ibcr = cv2.bitwise_and(Cr, Is1)

    # (b) 과정: Ib의 각 채널을 기존의 R, G, B 채널에 대응시켜 함수 'f'를 적용합니다.
    # 'f' 함수의 구체적인 작업은 문제에서 설명하지 않았으므로 여기서는 단순 복사를 가정합니다.
    f_R = Iby
    f_G = Ibcb
    f_B = Ibcr

    # (c) 과정: f_R, f_G, f_B 채널과 Is1을 AND 연산합니다.
    inverse_image = cv2.bitwise_not(Is1)
    cv2.imwrite(f'00.Results/00.Y_Cb_Cr/{idx}_Y_CB_CR_03inverse.png', inverse_image)
    Ib_R = cv2.bitwise_and(f_R, Is1)
    Ib_G = cv2.bitwise_and(f_G, Is1)
    Ib_B = cv2.bitwise_and(f_B, Is1)

    orginal_R, orginal_G, orginal_B = cv2.split(orginal)
    R = cv2.bitwise_and(orginal_R, inverse_image)
    G = cv2.bitwise_and(orginal_G, inverse_image)
    B = cv2.bitwise_and(orginal_B, inverse_image)

    # 최종 결과 이미지를 생성합니다.
    final_image = cv2.merge([Ib_R, Ib_G, Ib_B])
    seg_img = cv2.merge([R, G, B])

    cv2.imwrite(f'00.Results/00.Y_Cb_Cr/{idx}_Y_CB_CR_04final.png', final_image)
    cv2.imwrite(f'00.Results/00.Y_Cb_Cr/{idx}_Y_CB_CR_05Seg.png', seg_img)

if __name__ == "__main__":
    img_path = '../01.Dataset/AIHUB-persimmon/1.Training/*/*'
    load_image(img_path)
