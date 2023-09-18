# ## Handles all the preprocessing
# import numpy as np	# 벡터 및 행렬 연산
# from itertools import product	# itertools : 순열, 조합, product 구현,사용
# 				# poduct : 데카르트 곱 (cartesian product) = 2개 이상의 리스트의 모든 조합 구함
#
# inf = float('inf')	# 그 자체로 ∞를 의미
#
# #https://wh00300.tistory.com/204
# def rasterScan(image, blocksize, step=None):    # raster scan order : 데이터를 스캐닝 하는 방법
#                                                 # 좌측상단 -> 첫 라인부터 스캔 -> vertical 옮김 (한줄 아래왼쪽부터) -> 다음 줄 스캔
#                                                 # 한 블록 단위로 , minus the overlap
# 	'''
# 	Perform raster scan for image with squared block size "b"
# 	- If block size is not divisible(나눌 수 있는) by image size, then take all except last block
# 	- And for the last block, take the block from the other end
# 	'''
# 	block_list = []
# 	if step is None:
# 		step = blocksize
#
# 	H, W = image.shape[:2]	# opencv -> height, weight값 넣어줌
# 	Y = range(0, H-blocksize, step)	# 이미지 상 세로(y축) : 0, blocksize, blocksize*2, ... , Height-blocksize-blocksize
# 														# (2번재 인자 전 까지 - 포함 x)
# 	X = range(0, W-blocksize, step)	# 이미지 상 가로(x축) : 0, blocksize, blocksize*2, ... , Width-blocksize-blocksize
# 	if H%step != 0:	# Height 가 blocksize 로 나눠떨어지지 않을 경우
# 		Y = Y[:-1]	# 처음부터 ~ 맨 마지막 값 제외한 전부
# 	if W%step != 0:	# Width 가 blocksize 로 나눠떨어지지 않을 경우
# 		X = X[:-1]	# 처음부터 ~ 맨 마지막값 전까지
#
# 	for y in Y:	# 2) 다음 세로로 내려감
# 		for x in X:	# 1) 가로 부터 훑음
# 			block_list.append(image[y:y+blocksize, x:x+blocksize, :])	# block_list 에 append
# 																		# opencv -> image : (y크기(height),x크기(width),색상 채널)(픽셀)
# 								 										# image 에서 정해진 사이즈의 블록들 값정보 구분 , 색은 전체
#
# 	print("Created {} blocks.".format(len(block_list)))	# format : 문자열 포메팅 , 문자열 중간중간에 특정 변수값 넣기위해 사용
# 														# 사용 :{인덱스0}, {인덱스1}'.format(값0, 값1)
# 	return block_list	# image에서 블록들 나눠서 각각 값 저장한 리스트
#
# # def VerticalOverlap(im1, im2, blocksize, overlap):	# 옆으로 겹치는 오버랩
# # 	'''
# # 	Horizontal overlap between im1 (left) and im2 (right)
# # 	'''
# # 	im1Rot = np.rot90(im1)	# input한 배열을 반시계방향 90도 회전을 몇회 해줄 것인가? (default : 1회)
# # 	im2Rot = np.rot90(im2)	# input한 배열을 반시계방향 90도 회전을 몇회 해줄 것인가? (default : 1회)
# #
# # 	# mask : 넘파이 배열 -> 비교연산자 등 실행 -> True/False 논리값 가지는 배열
# # 	mask, minVal = HorizontalOverlap(im1Rot, im2Rot, blocksize, overlap)	# 함수 어디 정의??
# # 	mask = np.rot90(mask, 3)	# 배열 mask를 반시계방향 90도 회전 3회 : 270도 = 시계방향 90도 회전 1회
# #
# # 	# plt.imshow(mask)
# # 	# plt.show()
# # 	return mask, minVal
