import os	# 파이썬 기본 내제 모율, 운영체제와 상호작용을 돕는 다양한 기능 제공 (디렉토리 확인변경, csv 파일 호출 ..)
import numpy as np	# 벡터 및 행렬 연산
import cv2	# OpenCV import -> 오픈소스 컴퓨터 비전 및 머신러닝 라이브러리
			# 먼저 설치 : pip install opencv-python
from matplotlib import pyplot as plt	# matplotlib : 자료를 시각화 하는데 사용하는 대표 라이브러리 , 그래프 등 그림
from math import ceil	# math 함수 -> ceil : 올림 (int형)
from itertools import product	# itertools : 순열, 조합, product 구현,사용
				# poduct : 데카르트 곱 (cartesian product) = 2개 이상의 리스트의 모든 조합 구함


inf = float('inf')	# 그 자체로 ∞를 의미
ErrorCombinationFunc = np.add	# ?? np.add: array 요소 단위로 덧셈 계산..


def findPatchHorizontal(refBlock, texture, blocksize, overlap, tolerance):	# tolerance : 허용오차
	'''
	Find best horizontal match from the texture
	'''
	H, W = texture.shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
								# openCV 경우 -> (rows, columns, channels) 튜플 보유
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf	# np.zeros : 0으로 채워진 array 생성 / [[W-blocksize 만큼]*H-blocksize만큼] 0으로된 2차원 배열 생성
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# [0,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		rmsVal = ((texture[i:i+blocksize, j:j+overlap] - refBlock[:, -overlap:])**2).mean()	# (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균
		if rmsVal > 0:
			errMat[i, j] = rmsVal	# 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	minVal = np.min(errMat)	# 에러범위 값 중 가장 작은 것
	y, x = np.where(errMat < (1.0 + tolerance)*(minVal))	# np.where: 조건에 맞는 위치 인덱스 찾기 / 해당 허용오차보다 작은 E고름
															# y : [뽑힌 원소 각각 행 어디인지]
															# x : [뽑힌 원소 각각 열 어디인지] - (y,x) 둘이 이어서 위치 찾기
	c = np.random.randint(len(y))	# random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
	y, x = y[c], x[c]	# 허용오차 안의 해당 에러중 랜덤하게 뽑음
	return texture[y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return


def findPatchBoth(refBlockLeft, refBlockTop, texture, blocksize, overlap, tolerance):
	'''
	Find best horizontal and vertical match from the texture
	'''
	H, W = texture.shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
								# openCV 경우 -> (rows, columns, channels) 튜플 보유
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf	# np.zeros : 0으로 채워진 array 생성 / [[W-blocksize 만큼]*H-blocksize만큼] 0으로된 2차원 배열 생성
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# [0,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		rmsVal = ((texture[i:i+overlap, j:j+blocksize] - refBlockTop[-overlap:, :])**2).mean()	# (위의 이웃 블록의 오버랩 부분 - 각 블록의 위쪽 오버랩 부분) 제곱 의 평균
		rmsVal = rmsVal + ((texture[i:i+blocksize, j:j+overlap] - refBlockLeft[:, -overlap:])**2).mean()	# (왼쪽의 이웃 블록의 오버랩 부분 - 각 블록의 오른쪽 오버랩 부분) 제곱 의 평균
		if rmsVal > 0:
			errMat[i, j] = rmsVal	# 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	minVal = np.min(errMat)	# 에러범위 값 중 가장 작은 것
	y, x = np.where(errMat < (1.0 + tolerance)*(minVal))	# np.where: 조건에 맞는 위치 인덱스 찾기 / 해당 허용오차보다 작은 E고름
															# y : [뽑힌 원소 각각 행 어디인지]
															# x : [뽑힌 원소 각각 열 어디인지] - (y,x) 둘이 이어서 위치 찾기
	c = np.random.randint(len(y))	# random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
	y, x = y[c], x[c]	# 허용오차 안의 해당 에러중 랜덤하게 뽑음
	return texture[y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return


def findPatchVertical(refBlock, texture, blocksize, overlap, tolerance):
	'''
	Find best vertical match from the texture
	'''
	H, W = texture.shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
								# openCV 경우 -> (rows, columns, channels) 튜플 보유
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf	# np.zeros : 0으로 채워진 array 생성 / [[W-blocksize 만큼]*H-blocksize만큼] 0으로된 2차원 배열 생성
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# [0,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		rmsVal = ((texture[i:i+overlap, j:j+blocksize] - refBlock[-overlap:, :])**2).mean()	# (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균
		if rmsVal > 0:
			errMat[i, j] = rmsVal	# 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	minVal = np.min(errMat)	# 에러범위 값 중 가장 작은 것
	y, x = np.where(errMat < (1.0 + tolerance)*(minVal))	# np.where: 조건에 맞는 위치 인덱스 찾기 / 해당 허용오차보다 작은 E고름
															# y : [뽑힌 원소 각각 행 어디인지]
															# x : [뽑힌 원소 각각 열 어디인지] - (y,x) 둘이 이어서 위치 찾기
	c = np.random.randint(len(y))	# random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
	y, x = y[c], x[c]	# 허용오차 안의 해당 에러중 랜덤하게 뽑음
	return texture[y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return


def getMinCutPatchHorizontal(block1, block2, blocksize, overlap):
	'''
	Get the min cut patch done horizontally
	'''
	err = ((block1[:, -overlap:] - block2[:, :overlap])**2).mean(2)	# mean(2)?? / ((두 블록의 오버랩 부분의 차) 제곱 ) 평균
	# maintain minIndex for 2nd row onwards and
	# -> E 구하는건 윗 행의 E 값들의 min 을 비교하기 때문에 두번째 행의 E 부터 가능하다는 뜻으로 이해
	minIndex = []
	E = [list(err[0])]	# 첫 행만 뽑아서 리스트로 만듬
						# [[0.0032859156734589266, 0.004070229398949124, 0.003967704728950406]] 형태

	for i in range(1, err.shape[0]):	# .shape[0] : 행의 개수
		# Get min values and args, -1 = left, 0 = middle, 1 = right
		e = [inf] + E[-1] + [inf]	# [inf, 0.0032859156734589266, 0.004070229398949124, 0.003967704728950406, inf] 형태
		e = np.array([e[:-2], e[1:-1], e[2:]])	# 배열 생성
												# [[       inf 0.04719211 0.04602845]
 												# [0.04719211 0.04602845 0.04679739]
     											# [0.04602845 0.04679739        inf]] 형태

		# Get minIndex
		minArr = e.min(0)	# e에서의 각 행의 최소 값 / [첫번째 행 최소 값, 두번째 행 최소값, 세번째 행 최소값]
		minArg = e.argmin(0) - 1	# e.argmin : e에서 각 행의 최소값 위치 반환 -> [0,1,2] -1 -> [-1 , 0 , 1] / [첫번째 행 최소 값 위치, 두번째 행 최소값 위치, 세번째 행 최소값 위치]
		minIndex.append(minArg)	# 최소 위치 계속 추가 -> [1 1 0] [0 1 0] [1 0 -1] + + +
		print(minIndex)
		# Set Eij = e_ij + min_
		Eij = err[i] + minArr	# minArr = min( E(i-1 , j-1) , E(i-1 , j), E(i-1 , j+1) )
		E.append(list(Eij))	# 식을 이용해 구한 err 값 첫 행만 뽑은 E에 추가

	# Check the last element and backtrack to find path
	path = []
	minArg = np.argmin(E[-1])	# 전체적으로 overlap 부분 오류에 대하여 모두 구한 E 모두에 젤 마지막 줄의 최소값의 위치 (ex) 0)
	path.append(minArg)	# path 에 넣어줌


	# Backtrack to min path
	for idx in minIndex[::-1]:	# 전체적으로 E 구할 때 2번째 행부터 마지막까지 한 행 당 [inf A B C] [A B C] [B C inf] 에 대하여 뒤에서부터 보는 배열 / [0 1 1] [-1 0 1] ...

		minArg = minArg + idx[minArg]	# 이해 필요?? /
		path.append(minArg)

	# Reverse to find full path
	path = path[::-1]	# 거꾸로 출력
	mask = np.zeros((blocksize, blocksize, block1.shape[2]))	# 0으로된 배열 생성 / .shape[2]: 컬러채널 / 3차원 [블록사이즈, 블록사이즈 , 컬러]
	for i in range(len(path)):
		mask[i, :path[i]+1] = 1	# path[i]+1] : 위치 값 넣어줄 때 -1 했었으므로 다시 0,1,2 형태로 만들기 위해서 1 더해줌

	resBlock = np.zeros(block1.shape)	# 블록과 같은 형태 0으로된 배열 -> 새로 끼울 블록 -> 왼쪽 경계 울퉁불퉁하게 만들어야함
	resBlock[:, :overlap] = block1[:, -overlap:]	# 왼쪽 처음에서 부터 overlap 부분까지 -> 기존 블록인 블록 1의 오른쪽 오버랩 부분을 대입
	resBlock = resBlock*mask + block2*(1-mask)	# 최종 새로운 블록 왼쪽 경계 = path 넣어준 마스크 곱함 + 새로 끼울 블록 2도 그 반대로 마스크가 생성되어 곱함
	# resBlock = block1*mask + block2*(1-mask)
	return resBlock


def getMinCutPatchVertical(block1, block2, blocksize, overlap):
	'''
	Get the min cut patch done vertically
	'''
	resBlock = getMinCutPatchHorizontal(np.rot90(block1), np.rot90(block2), blocksize, overlap)
	return np.rot90(resBlock, 3)


def getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap):
	'''
	Find minCut for both and calculate
	'''
	err = ((refBlockLeft[:, -overlap:] - patchBlock[:, :overlap])**2).mean(2)
	# maintain minIndex for 2nd row onwards and 
	minIndex = []
	E = [list(err[0])]
	for i in range(1, err.shape[0]):
		# Get min values and args, -1 = left, 0 = middle, 1 = right
		e = [inf] + E[-1] + [inf]
		e = np.array([e[:-2], e[1:-1], e[2:]])
		# Get minIndex
		minArr = e.min(0)
		minArg = e.argmin(0) - 1
		minIndex.append(minArg)
		# Set Eij = e_ij + min_
		Eij = err[i] + minArr
		E.append(list(Eij))

	# Check the last element and backtrack to find path
	path = []
	minArg = np.argmin(E[-1])
	path.append(minArg)

	# Backtrack to min path
	for idx in minIndex[::-1]:
		minArg = minArg + idx[minArg]
		path.append(minArg)
	# Reverse to find full path
	path = path[::-1]
	mask1 = np.zeros((blocksize, blocksize, patchBlock.shape[2]))
	for i in range(len(path)):
		mask1[i, :path[i]+1] = 1

	###################################################################
	## Now for vertical one
	err = ((np.rot90(refBlockTop)[:, -overlap:] - np.rot90(patchBlock)[:, :overlap])**2).mean(2)
	# maintain minIndex for 2nd row onwards and 
	minIndex = []
	E = [list(err[0])]
	for i in range(1, err.shape[0]):
		# Get min values and args, -1 = left, 0 = middle, 1 = right
		e = [inf] + E[-1] + [inf]
		e = np.array([e[:-2], e[1:-1], e[2:]])
		# Get minIndex
		minArr = e.min(0)
		minArg = e.argmin(0) - 1
		minIndex.append(minArg)
		# Set Eij = e_ij + min_
		Eij = err[i] + minArr
		E.append(list(Eij))

	# Check the last element and backtrack to find path
	path = []
	minArg = np.argmin(E[-1])
	path.append(minArg)

	# Backtrack to min path
	for idx in minIndex[::-1]:
		minArg = minArg + idx[minArg]
		path.append(minArg)
	# Reverse to find full path
	path = path[::-1]
	mask2 = np.zeros((blocksize, blocksize, patchBlock.shape[2]))
	for i in range(len(path)):
		mask2[i, :path[i]+1] = 1
	mask2 = np.rot90(mask2, 3)


	mask2[:overlap, :overlap] = np.maximum(mask2[:overlap, :overlap] - mask1[:overlap, :overlap], 0)

	# Put first mask
	resBlock = np.zeros(patchBlock.shape)
	resBlock[:, :overlap] = mask1[:, :overlap]*refBlockLeft[:, -overlap:]
	resBlock[:overlap, :] = resBlock[:overlap, :] + mask2[:overlap, :]*refBlockTop[-overlap:, :]
	resBlock = resBlock + (1-np.maximum(mask1, mask2))*patchBlock
	return resBlock




def generateTextureMap(image, blocksize, overlap, outH, outW, tolerance):
	nH = int(ceil((outH - blocksize)*1.0/(blocksize - overlap)))
	nW = int(ceil((outW - blocksize)*1.0/(blocksize - overlap)))

	textureMap = np.zeros(((blocksize + nH*(blocksize - overlap)), (blocksize + nW*(blocksize - overlap)), image.shape[2]))
	
	# Starting index and block
	H, W = image.shape[:2]
	randH = np.random.randint(H - blocksize)
	randW = np.random.randint(W - blocksize)

	startBlock = image[randH:randH+blocksize, randW:randW+blocksize]
	textureMap[:blocksize, :blocksize, :] = startBlock

	# Fill the first row 
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[1]-overlap, (blocksize-overlap))):
		# Find horizontal error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		refBlock = textureMap[:blocksize, (blkIdx-blocksize+overlap):(blkIdx+overlap)]
		patchBlock = findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance)
		minCutPatch = getMinCutPatchHorizontal(refBlock, patchBlock, blocksize, overlap)
		textureMap[:blocksize, (blkIdx):(blkIdx+blocksize)] = minCutPatch
	print("{} out of {} rows complete...".format(1, nH+1))


	### Fill the first column
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[0]-overlap, (blocksize-overlap))):
		# Find vertical error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		refBlock = textureMap[(blkIdx-blocksize+overlap):(blkIdx+overlap), :blocksize]
		patchBlock = findPatchVertical(refBlock, image, blocksize, overlap, tolerance)
		minCutPatch = getMinCutPatchVertical(refBlock, patchBlock, blocksize, overlap)

		textureMap[(blkIdx):(blkIdx+blocksize), :blocksize] = minCutPatch

	### Fill in the other rows and columns
	for i in range(1, nH+1):
		for j in range(1, nW+1):
			# Choose the starting index for the texture placement
			blkIndexI = i*(blocksize-overlap)
			blkIndexJ = j*(blocksize-overlap)
			# Find the left and top block, and the min errors independently
			refBlockLeft = textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ-blocksize+overlap):(blkIndexJ+overlap)]
			refBlockTop  = textureMap[(blkIndexI-blocksize+overlap):(blkIndexI+overlap), (blkIndexJ):(blkIndexJ+blocksize)]

			patchBlock = findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance)
			minCutPatch = getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap) 

			textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ):(blkIndexJ+blocksize)] = minCutPatch

			# refBlockLeft = 0.5
			# textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ-blocksize+overlap):(blkIndexJ+overlap)] = refBlockLeft
			# textureMap[(blkIndexI-blocksize+overlap):(blkIndexI+overlap), (blkIndexJ):(blkIndexJ+blocksize)] = [0.5, 0.6, 0.7]
			# break
		print("{} out of {} rows complete...".format(i+1, nH+1))
		# break

	return textureMap
