import os	# 파이썬 기본 내제 모율, 운영체제와 상호작용을 돕는 다양한 기능 제공 (디렉토리 확인변경, csv 파일 호출 ..)
import numpy as np	# 벡터 및 행렬 연산
import cv2	# OpenCV import -> 오픈소스 컴퓨터 비전 및 머신러닝 라이브러리
			# 먼저 설치 : pip install opencv-python
from matplotlib import pyplot as plt	# matplotlib : 자료를 시각화 하는데 사용하는 대표 라이브러리 , 그래프 등 그림
from math import ceil	# math 함수 -> ceil : 올림 (int형)
import math	# 추가 : 사인 코사인 탄젠트 사용을 위해
from itertools import product	# itertools : 순열, 조합, product 구현,사용
				# poduct : 데카르트 곱 (cartesian product) = 2개 이상의 리스트의 모든 조합 구함
import  imageio # gif 파일 만들기 위한 라이브러리


inf = float('inf')	# 그 자체로 ∞를 의미
ErrorCombinationFunc = np.add	# ?? np.add: array 요소 단위로 덧셈 계산..

# original
def findPatchHorizontal(refBlock, texture, blocksize, overlap, tolerance):	# tolerance : 허용오차
	'''
	Find best horizontal match from the texture
	사용: findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance)
	'''
	H, W = texture.shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf	# np.zeros : 0으로 채워진 array 생성 / [[W-blocksize 만큼]*H-blocksize만큼] 0으로된 2차원 배열 생성 / inf 는 왜더하는지는 모르겠음???
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# openCV 경우 -> (rows, columns, channels) 튜플 보유,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		rmsVal = ((texture[i:i+blocksize, j:j+overlap] - refBlock[:, -overlap:])**2).mean()	# (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균
		if rmsVal > 0:
			errMat[i, j] = rmsVal	# 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	minVal = np.min(errMat)	# 에러범위 값 중 가장 작은 것
	y, x = np.where(errMat < (1.0 + tolerance)*(minVal))	# np.where: 조건에 맞는 위치 인덱스 찾기 / 해당 허용오차보다 작은 E 고름
															# y : [뽑힌 원소 각각 행 어디인지]
															# x : [뽑힌 원소 각각 열 어디인지] - (y,x) 둘이 이어서 위치 찾기
	c = np.random.randint(len(y))	# random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
	y, x = y[c], x[c]	# 허용오차 안의 해당 에러 중 랜덤하게 뽑음

	return texture[y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return


# original
def findPatchBoth(refBlockLeft, refBlockTop, texture, blocksize, overlap, tolerance):
	'''
	Find best horizontal and vertical match from the texture
	사용: findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance)
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


# original
def findPatchVertical(refBlock, texture, blocksize, overlap, tolerance):
	'''
	Find best vertical match from the texture
	사용: findPatchVertical(refBlock, image, blocksize, overlap, tolerance)
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


# original
def getMinCutPatchHorizontal(block1, block2, blocksize, overlap):
	'''
	Get the min cut patch done horizontally
	사용: getMinCutPatchHorizontal(refBlock, patchBlock, blocksize, overlap)
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
							# 형태 : [0.22153531 0.12927848 0.12927848]
		minArg = e.argmin(0) - 1	# e.argmin : e에서 각 행의 최소값 위치 반환 -> [0,1,2] -1 -> [-1 , 0 , 1] / [첫번째 행 최소 값 위치, 두번째 행 최소값 위치, 세번째 행 최소값 위치]
		minIndex.append(minArg)	# 최소 위치 계속 추가 -> [1 1 0] [0 1 0] [1 0 -1] + + +
		# Set Eij = e_ij + min_
		Eij = err[i] + minArr	# minArr = min( E(i-1 , j-1) , E(i-1 , j), E(i-1 , j+1) )
		E.append(list(Eij))	# 식을 이용해 구한 err 값 첫 행만 뽑은 E에 추가
		# E 행개수 : n개 , minIndex 행개수 : n-1개
	# E 형태 :  [[0.01045751633986928, 0.02260156350121748, 0.008145585031398181], ... ,[0.376080994489299, 0.37763936947327953, 0.3876560297321543]]

	# Check the last element and backtrack to find path
	path = []
	minArg = np.argmin(E[-1])	# np.argmin: 최소값 인덱스 반환
								# 전체적으로 overlap 부분 오류에 대하여 모두 구한 E 모두에 젤 마지막 줄의 최소값의 위치 (ex) 0) / 0,1,2 중 최소인덱스 하나 입력됨
	path.append(minArg)	# path 에 넣어줌


	# Backtrack to min path
	for idx in minIndex[::-1]:	# 전체적으로 E 구할 때 2번째 행부터 마지막까지 한 행 당 [inf A B C] [A B C] [B C inf] 에 대하여 뒤에서부터 보는 배열 / [0 1 1] [-1 0 1] ...

		minArg = minArg + idx[minArg]	# 이해 필요??  / -1,0,1,2 중 하나의 값 나옴
										# idx 로 움직이며 , 이전 minArg 값으로 다음 값을 갱신함
		path.append(minArg)

	# Reverse to find full path
	path = path[::-1]	# 거꾸로 출력
	mask = np.zeros((blocksize, blocksize, block1.shape[2]))	# 0으로된 배열 생성 / .shape[2]: 컬러채널 / 3차원 [블록사이즈, 블록사이즈 , 컬러]
	for i in range(len(path)):
		mask[i, :path[i]+1] = 1	# path[i]+1] : 위치 값 넣어줄 때 -1 했었으므로 다시 0,1,2 형태로 만들기 위해서 1 더해줌

	resBlock = np.zeros(block1.shape)	# 블록과 같은 형태 0으로된 배열 -> 새로 끼울 블록 -> 왼쪽 경계 울퉁불퉁하게 만들어야함
	resBlock[:, :overlap] = block1[:, -overlap:]	# 왼쪽 처음에서 부터 overlap 부분까지 -> 기존  블록인 블록 1의 오른쪽 오버랩 부분을 대입
	resBlock = resBlock*mask + block2*(1-mask)	# 최종 새로운 블록 왼쪽 경계 = path 넣어준 마스크 곱함 + 새로 끼울 블록 2도 그 반대로 마스크가 생성되어 곱함
	# resBlock = block1*mask + block2*(1-mask)
	return resBlock

# original
def getMinCutPatchVertical(block1, block2, blocksize, overlap):	# horizontal 에서 인자 블록만 반시계 90도 돌려서 같은 계산 행함
	'''
	Get the min cut patch done vertically
	'''
	resBlock = getMinCutPatchHorizontal(np.rot90(block1), np.rot90(block2), blocksize, overlap)	# np.rot90 : 반시계방향으로 90도 회전 / horizontal 과 동일해짐
	return np.rot90(resBlock, 3)	# 이미 90도 돌린 상태로 계산 -> 원상복귀 -> 270 도 더 돌려서 360도 만들어줌


def getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap):
	'''
	Find minCut for both and calculate
	'''
	err = ((refBlockLeft[:, -overlap:] - patchBlock[:, :overlap])**2).mean(2)	# mean(2)?? / ((왼쪽 기존블록과 새로운 패치블록의 오버랩 부분의 차) 제곱 ) 평균
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
							# 형태 : [0.22153531 0.12927848 0.12927848]
		minArg = e.argmin(0) - 1	# e.argmin : e에서 각 행의 최소값 위치 반환 -> [0,1,2] -1 -> [-1 , 0 , 1]
									# [첫번째 행 최소 값 위치, 두번째 행 최소값 위치, 세번째 행 최소값 위치]
		minIndex.append(minArg)	# 최소 위치 계속 추가 -> [1 1 0] [0 1 0] [1 0 -1] + + +
		# Set Eij = e_ij + min_
		Eij = err[i] + minArr	# minArr = min( E(i-1 , j-1) , E(i-1 , j), E(i-1 , j+1) )
		E.append(list(Eij))	# 식을 이용해 구한 err 값 첫 행만 뽑은 E에 추가
	# E 형태 :  [[0.01045751633986928, 0.02260156350121748, 0.008145585031398181], ... ,[0.376080994489299, 0.37763936947327953, 0.3876560297321543]]

	# Check the last element and backtrack to find path
	path = []
	minArg = np.argmin(E[-1])	# 전체적으로 overlap 부분 오류에 대하여 모두 구한 E 모두에 젤 마지막 줄의 최소값의 위치 (ex) 0) / 0,1,2 중 최소인덱스 하나 입력됨
	path.append(minArg)	# path 에 넣어줌

	# Backtrack to min path
	for idx in minIndex[::-1]:	# 전체적으로 E 구할 때 2번째 행부터 마지막까지 한 행 당 [inf A B C] [A B C] [B C inf] 에 대하여 뒤에서부터 보는 배열 / [0 1 1] [-1 0 1] ...
		minArg = minArg + idx[minArg]	# 이해 필요?? / -1,0,1,2 중 하나의 값 나옴
										# idx 로 움직이며 , 이전 minArg 값으로 다음 값을 갱신함
		path.append(minArg)

	# Reverse to find full path
	path = path[::-1]	# 거꾸로 출력
	# 마스크 만들기!
	mask1 = np.zeros((blocksize, blocksize, patchBlock.shape[2]))	# 0으로된 배열 생성 / .shape[2]: 컬러채널 / 3차원 [블록사이즈, 블록사이즈 , 컬러]
	for i in range(len(path)):
		mask1[i, :path[i]+1] = 1	# path[i]+1] : 위치 값 넣어줄 때 -1 했었으므로 다시 0,1,2 형태로 만들기 위해서 1 더해줌


	###################################################################
	## Now for vertical one -> horizontal 하고 똑같이
	err = ((np.rot90(refBlockTop)[:, -overlap:] - np.rot90(patchBlock)[:, :overlap])**2).mean(2)	# mean(2)?? / ((반시계방향으로 회전시킨 두 블록의 차) 제곱 ) 평균

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
	# 마스크 만들기!
	mask2 = np.zeros((blocksize, blocksize, patchBlock.shape[2]))	# 0으로된 배열 생성 / .shape[2]: 컬러채널 / 3차원 [블록사이즈, 블록사이즈 , 컬러]
	for i in range(len(path)):
		mask2[i, :path[i]+1] = 1	# path[i]+1] : 위치 값 넣어줄 때 -1 했었으므로 다시 0,1,2 형태로 만들기 위해서 1 더해줌
	mask2 = np.rot90(mask2, 3)	# 90도 반시계방향으로 돌려줬으므로 -> 원상복귀 -> 270 도 더 돌려서 360도로 만듬


	mask2[:overlap, :overlap] = np.maximum(mask2[:overlap, :overlap] - mask1[:overlap, :overlap], 0)	# np.maximum : 여러 array 사이에서 각 위치의 최대값
	# mask2 오버랩 제외부분 = [mask2 - mask1 의 각 최대값들의 배열0, mask2 - mask1 의 각 최대값들의 배열1, 0]

	# Put first mask
	resBlock = np.zeros(patchBlock.shape)
	resBlock[:, :overlap] = mask1[:, :overlap]*refBlockLeft[:, -overlap:]	# resBlock 에 오른쪽 오버랩 부분을 mask1 값도 곱해서 다시 대입
	resBlock[:overlap, :] = resBlock[:overlap, :] + mask2[:overlap, :]*refBlockTop[-overlap:, :]	# resBlock 에 위쪽 오버랩 부분을 mask2 값도 곱해서 다시 대입
	resBlock = resBlock + (1-np.maximum(mask1, mask2))*patchBlock	# 기존 패치블록도 마스크 반대값(반대로 울퉁불퉁 모양으로 자름) 해줘서 더함
	return resBlock

########################################

# original
def generateTextureMap(image, blocksize, overlap, outH, outW, tolerance):	# main.py에서 사용되는 메인. tolerance : 허용요차
	# 사용: generateTextureMap(image, block_size, overlap, outH, outW, args.tolerance)
	# ceil() : 소수점 자리의 숫자를 무조건 올리는 함수
	nH = int(ceil((outH - blocksize)*1.0/(blocksize - overlap)))	# 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?
	nW = int(ceil((outW - blocksize)*1.0/(blocksize - overlap)))	# 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?

	textureMap = np.zeros(((blocksize + nH*(blocksize - overlap)), (blocksize + nW*(blocksize - overlap)), image.shape[2]))
	# [(H기준 : nH(들어가는 블록개수) * (오버랩 뺀 블록실제사이즈) + 마지막에 오버랩 안되므로 블록 하나 더 사이즈) , (W기준 동일) , 색상] => 0으로 초기화
	# Starting index and block
	H, W = image.shape[:2]

	randH = np.random.randint(H - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값
	randW = np.random.randint(W - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값

	startBlock = image[randH:randH+blocksize, randW:randW+blocksize]	# 랜덤한 위치에서 시작하는 블록 사이즈만큼 잘라서 가져옴
	textureMap[:blocksize, :blocksize, :] = startBlock	# 0으로 초기화된 맵에서 첫번째 블록에 랜덤하게 가져온 블록 대입함

	# Fill the first row : 행(아래 위)
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[1]-overlap, (blocksize-overlap))):	# enumerate() : 인덱스와 원소 차례로 반환
		# 오버랩 부분 제외 블록 부분부터 ~ 오버랩 제외 열들까지 , 오버랩 제외한 블록사이즈만큼 옆으로 이동 (오른 -> 왼)

		# Find horizontal error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		# blkIdx = 블록에서 오버랩 되는 부분 시작점 인덱스
		refBlock = textureMap[:blocksize, (blkIdx-blocksize+overlap):(blkIdx+overlap)]	#texturemap 의 한줄제외 모든 행에 대하여 열단위로 블록 한 칸만큼 계속 이동하면서 대입
		patchBlock = findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance)	# 미리 만든 패치 찾는 함수
		minCutPatch = getMinCutPatchHorizontal(refBlock, patchBlock, blocksize, overlap)	# 미리 만든 최소 경로 찾는 함수
		textureMap[:blocksize, (blkIdx):(blkIdx+blocksize)] = minCutPatch	# 오버랩부분 경계선 최소경로로 자름
	print("{} out of {} rows complete...".format(1, nH+1))


	### Fill the first column 열 (오른 왼쪽)
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[0]-overlap, (blocksize-overlap))):	# # enumerate() : 인덱스와 원소 차례로 반환
		# 오버랩 부분 제외 블록 부분부터 ~ 오버랩 제외 행들까지 , 오버랩 제외한 블록사이즈만큼 옆으로 이동 (위 -> 아래)

		# Find vertical error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		# blkIdx = 블록에서 오버랩 되는 부분 시작점 인덱스
		refBlock = textureMap[(blkIdx-blocksize+overlap):(blkIdx+overlap), :blocksize]	#texturemap 의 한줄제외 모든 열에 대하여 행단위로 블록 한 칸만큼 계속 이동하면서 대입
		patchBlock = findPatchVertical(refBlock, image, blocksize, overlap, tolerance)	# 미리 만든 패치 찾는 함수
		minCutPatch = getMinCutPatchVertical(refBlock, patchBlock, blocksize, overlap)	# 미리 만든 최소 경로 찾는 함수
		textureMap[(blkIdx):(blkIdx+blocksize), :blocksize] = minCutPatch	# 오버랩부분 경계선 최소경로로 자름

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
#####################################################

#추가#########################
# 8회전시 합성부분
def r_findPatchHorizontal(refBlock, texture, blocksize, overlap, tolerance, mask):	# tolerance : 허용오차
	'''
	Find best horizontal match from the texture
	사용: findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance)
	'''
	H, W = texture.shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf	# np.zeros : 0으로 채워진 array 생성 / [[W-blocksize 만큼]*H-blocksize만큼] 0으로된 2차원 배열 생성 / inf 는 왜더하는지는 모르겠음???
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# openCV 경우 -> (rows, columns, channels) 튜플 보유,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)

		if (mask[i:i + blocksize, j:j + blocksize] == 1).all():
			rmsVal = ((texture[i:i + blocksize, j:j + overlap] - refBlock[:, -overlap:]) ** 2).mean()  # (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균
			if rmsVal > 0:
				errMat[i, j] = rmsVal  # 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	minVal = np.min(errMat)	# 에러범위 값 중 가장 작은 것
	y, x = np.where(errMat < (1.0 + tolerance)*(minVal))	# np.where: 조건에 맞는 위치 인덱스 찾기 / 해당 허용오차보다 작은 E 고름
															# y : [뽑힌 원소 각각 행 어디인지]
															# x : [뽑힌 원소 각각 열 어디인지] - (y,x) 둘이 이어서 위치 찾기

	while (True):
		c = np.random.randint(len(y))	# random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
		yy, xx = y[c], x[c]	# 허용오차 안의 해당 에러 중 랜덤하게 뽑음
		if (mask[yy:yy+blocksize, xx:xx+blocksize]==1).all():
			break

	return texture[yy:yy+blocksize, xx:xx+blocksize]	# 텍스쳐에서 해당 블록 return


# 8회전시 합성부분
def r_findPatchBoth(refBlockLeft, refBlockTop, texture, blocksize, overlap, tolerance, mask):
	'''
	Find best horizontal and vertical match from the texture
	사용: findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance)
	'''
	H, W = texture.shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
								# openCV 경우 -> (rows, columns, channels) 튜플 보유
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf	# np.zeros : 0으로 채워진 array 생성 / [[W-blocksize 만큼]*H-blocksize만큼] 0으로된 2차원 배열 생성
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# [0,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		if (mask[i:i+blocksize, j:j+blocksize] == 1).all():
			rmsVal = ((texture[i:i+overlap, j:j+blocksize] - refBlockTop[-overlap:, :])**2).mean()	# (위의 이웃 블록의 오버랩 부분 - 각 블록의 위쪽 오버랩 부분) 제곱 의 평균
			rmsVal = rmsVal + ((texture[i:i+blocksize, j:j+overlap] - refBlockLeft[:, -overlap:])**2).mean()	# (왼쪽의 이웃 블록의 오버랩 부분 - 각 블록의 오른쪽 오버랩 부분) 제곱 의 평균
			if rmsVal > 0:
				errMat[i, j] = rmsVal	# 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	minVal = np.min(errMat)	# 에러범위 값 중 가장 작은 것
	y, x = np.where(errMat < (1.0 + tolerance)*(minVal))	# np.where: 조건에 맞는 위치 인덱스 찾기 / 해당 허용오차보다 작은 E고름
															# y : [뽑힌 원소 각각 행 어디인지]
															# x : [뽑힌 원소 각각 열 어디인지] - (y,x) 둘이 이어서 위치 찾기
	while (True):
		c = np.random.randint(len(y))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
		yy, xx = y[c], x[c]  # 허용오차 안의 해당 에러 중 랜덤하게 뽑음
		if (mask[yy:yy + blocksize, xx:xx + blocksize] == 1).all():
			break

	return texture[yy:yy+blocksize, xx:xx+blocksize]	# 텍스쳐에서 해당 블록 return


# 8회전시 합성부분
def r_findPatchVertical(refBlock, texture, blocksize, overlap, tolerance, mask):
	'''
	Find best vertical match from the texture
	사용: findPatchVertical(refBlock, image, blocksize, overlap, tolerance)
	'''
	H, W = texture.shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
								# openCV 경우 -> (rows, columns, channels) 튜플 보유
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf	# np.zeros : 0으로 채워진 array 생성 / [[W-blocksize 만큼]*H-blocksize만큼] 0으로된 2차원 배열 생성
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# [0,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		if (mask[i:i+blocksize, j:j+blocksize] == 1).all():
			rmsVal = ((texture[i:i+overlap, j:j+blocksize] - refBlock[-overlap:, :])**2).mean()	# (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균
			if rmsVal > 0:
				errMat[i, j] = rmsVal	# 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	minVal = np.min(errMat)	# 에러범위 값 중 가장 작은 것
	y, x = np.where(errMat < (1.0 + tolerance)*(minVal))	# np.where: 조건에 맞는 위치 인덱스 찾기 / 해당 허용오차보다 작은 E고름
															# y : [뽑힌 원소 각각 행 어디인지]
															# x : [뽑힌 원소 각각 열 어디인지] - (y,x) 둘이 이어서 위치 찾기

	while (True):
		c = np.random.randint(len(y))	# random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
		yy, xx = y[c], x[c]	# 허용오차 안의 해당 에러 중 랜덤하게 뽑음
		if (mask[yy:yy+blocksize, xx:xx+blocksize]==1).all():
			break

	return texture[yy:yy+blocksize, xx:xx+blocksize]	# 텍스쳐에서 해당 블록 return
#############################

#추가#########################
# tan 추가 합성
def t_findPatchHorizontal(refBlock, texture, blocksize, overlap, tolerance, mask, blkIdx, where_white, where_black):	# tolerance : 허용오차
	'''
	Find best horizontal match from the texture
	사용: findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance)
	'''
	H, W = texture.shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf	# np.zeros : 0으로 채워진 array 생성 / [[W-blocksize 만큼]*H-blocksize만큼] 0으로된 2차원 배열 생성 / inf 는 왜더하는지는 모르겠음???
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# openCV 경우 -> (rows, columns, channels) 튜플 보유,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		rmsVal = ((texture[i:i+blocksize, j:j+overlap] - refBlock[:, -overlap:])**2).mean()	# (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균

		if rmsVal > 0:
			errMat[i, j] = rmsVal	# 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	print("err[]: {}".format(errMat))

	#minVal = np.min(errMat)  # 에러범위 값 중 가장 작은 것
	y = 0
	x = 0

	if (mask[:blocksize, (blkIdx):(blkIdx + blocksize)] == 1).any():  # 마스크 안에 들어갈 경우

		minVal = 1000
		c = 0
		print(len(where_white))
		for ii in range(len(where_white)):
			if errMat[where_white[ii][0], where_white[ii][1]] < minVal:
				print("block err:{}".format(errMat[where_white[ii][0], where_white[ii][1]]))
				minVal = errMat[where_white[ii][0], where_white[ii][1]]
				c = ii

		where_white2=[]
		for k in range(len(where_white)):	# foam data 일정이상 하얀부분인 인덱스 중
			if errMat[where_white[k][0],where_white[k][1]] < (1.0 + tolerance)*(minVal):	# err가 허용오차 값 이하일때의 인덱스 따로 빼둠
				where_white2.append(where_white[k])
		if len(where_white2)==0:	# 하얀부분의 해당위치의 err가 허용오차 값 이하인 블록이 존재하지 않을 때
			for k in range(len(where_white)):  # foam data 일정이상 하얀부분인 인덱스 중
				if errMat[where_white[k][0], where_white[k][1]] < (1.0 + 0.5) * (minVal):  # 허용 오차 늘려서 에러 다시 구함
					where_white2.append(where_white[k])
			if len(where_white2)==0:
				for k in range(len(where_white)):  # foam data 일정이상 하얀부분인 인덱스 중
					where_white2.append(where_white[k])

			c = np.random.randint(len(where_white2))
			y = where_white2[c][0]
			x = where_white2[c][1]
		else:
			c = np.random.randint(len(where_white))
			y = where_white[c][0]
			x = where_white[c][1]
	else:	# 마스크 밖 부분

		minVal = 1000
		c = 0
		for ii in range(len(where_black)):
			if errMat[where_black[ii][0], where_black[ii][1]] < minVal:
				minVal = errMat[where_black[ii][0], where_black[ii][1]]
				c = ii

		where_black2 = []
		for k in range(len(where_black)):  # foam data 일정이상 검은부분인 인덱스 중
			if errMat[where_black[k][0], where_black[k][1]] < (1.0 + tolerance) * (
					minVal):  # err가 허용오차 값 이하일때의 인덱스 따로 빼둠
				where_black2.append(where_black[k])
		if len(where_black2) == 0:  # 검은부분의 해당위치의 err가 허용오차 값 이하인 블록이 존재하지 않을 때
			for k in range(len(where_black)):  # foam data 일정이상 하얀부분인 인덱스 중
				if errMat[where_black[k][0], where_black[k][1]] < (1.0 + 0.5) * (minVal):  # 허용 오차 늘려서 에러 다시 구함

					where_black2.append(where_black[k])
			if len(where_black2)==0:
				for k in range(len(where_black)):  # foam data 일정이상 하얀부분인 인덱스 중
					where_black2.append(where_black[k])

			c = np.random.randint(len(where_black2))
			y = where_black2[c][0]
			x = where_black2[c][1]
		else:  # 검은부분의 해당위치의 err가 허용오차 값 이하인 블록이 존재할때 -> 랜덤하게
			c = np.random.randint(len(where_black2))
			y = where_black2[c][0]
			x = where_black2[c][1]

	# count = []
	# for i in range(len(y)):
	# 	tmp = 0
	# 	for j in range(y[i], y[i] + blocksize + 1):
	# 		for k in range(x[i], x[i] + blocksize + 1):
	# 			if (mask[:blocksize, (blkIdx):(blkIdx + blocksize)] == 1).any():	# 마스크 안에 들어갈 경우
	# 				if (texture[j, k] >= 0.7).all():
	# 					tmp += 1
	# 			else:
	# 				if (texture[j, k] < 0.3).all():
	# 					tmp += 1
	# 	count.append(tmp)
	#
	# c = count.index(max(count))

	# #############
	# c = np.random.randint(len(y))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
	# y, x = y[c], x[c]  # 허용오차 안의 해당 에러 중 랜덤하게 뽑음
	#
	# # for tmpH in range(H):
	# # 	for tmeW in range(W):
	#
	# return texture[y:y + blocksize, x:x + blocksize]  # 텍스쳐에서 해당 블록 return
	# #############
	# count = []
	# for i in range(len(y)):
	# 	tmp = 0
	# 	for j in range(y[i], y[i] + blocksize + 1):
	# 		for k in range(x[i], x[i] + blocksize + 1):
	# 			if (texture[j, k] >= 0.3).all():
	# 				tmp += 1
	# 	count.append(tmp)
	#
	# if (mask[:blocksize, (blkIdx):(blkIdx + blocksize)]==1).any():
	# 	c = count.index(max(count))
	# else:
	# 	c = count.index(min(count))

	# y, x = y[c], x[c]  # 허용오차 안의 해당 에러 중 랜덤하게 뽑음

	return texture[y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return




# tan 추가 합성
def t_findPatchBoth(refBlockLeft, refBlockTop, texture, blocksize, overlap, tolerance, mask, blkIndexI, blkIndexJ, where_white, where_black):
	'''
	Find best horizontal and vertical match from the texture
	사용: findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance)
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

	#minVal = np.min(errMat)	# 에러범위 값 중 가장 작은 것
	# y, x = np.where(errMat < (1.0 + tolerance)*(minVal))	# np.where: 조건에 맞는 위치 인덱스 찾기 / 해당 허용오차보다 작은 E고름
	# 														# y : [뽑힌 원소 각각 행 어디인지]
	# 														# x : [뽑힌 원소 각각 열 어디인지] - (y,x) 둘이 이어서 위치 찾기

	y = 0
	x = 0

	if (mask[(blkIndexI):(blkIndexI + blocksize), (blkIndexJ):(blkIndexJ + blocksize)]==1).any():  # 마스크 안에 들어갈 경우


		minVal = 1000
		c = 0
		for ii in range(len(where_white)):
			if errMat[where_white[ii][0], where_white[ii][1]] < minVal:
				minVal = errMat[where_white[ii][0], where_white[ii][1]]
				c = ii

		where_white2 = []
		for k in range(len(where_white)):  # foam data 일정이상 하얀부분인 인덱스 중
			if errMat[where_white[k][0], where_white[k][1]] < (1.0 + tolerance) * (minVal):  # err가 허용오차 값 이하일때의 인덱스 따로 빼둠
				where_white2.append(where_white[k])
		if len(where_white2) == 0:  # 하얀부분의 해당위치의 err가 허용오차 값 이하인 블록이 존재하지 않을 때
			for k in range(len(where_white)):  # foam data 일정이상 하얀부분인 인덱스 중
				if errMat[where_white[k][0], where_white[k][1]] < (1.0 + 0.3) * (minVal):  # 허용 오차 늘려서 에러 다시 구함
					where_white2.append(where_white[k])
			if len(where_white2)==0:
				for k in range(len(where_white)):  # foam data 일정이상 하얀부분인 인덱스 중
					where_white2.append(where_white[k])

			c = np.random.randint(len(where_white2))
			y = where_white2[c][0]
			x = where_white2[c][1]
		else:
			c = np.random.randint(len(where_white))
			y = where_white[c][0]
			x = where_white[c][1]
	else:  # 마스크 밖 부분

		minVal = 1000
		c = 0
		for ii in range(len(where_black)):
			if errMat[where_black[ii][0], where_black[ii][1]] < minVal:
				minVal = errMat[where_black[ii][0], where_black[ii][1]]
				c = ii

		where_black2 = []
		for k in range(len(where_black)):  # foam data 일정이상 검은부분인 인덱스 중
			if errMat[where_black[k][0], where_black[k][1]] < (1.0 + tolerance) * (
			minVal):  # err가 허용오차 값 이하일때의 인덱스 따로 빼둠
				where_black2.append(where_black[k])
		if len(where_black2) == 0:  # 검은부분의 해당위치의 err가 허용오차 값 이하인 블록이 존재하지 않을 때
			for k in range(len(where_black)):  # foam data 일정이상 하얀부분인 인덱스 중
				if errMat[where_black[k][0], where_black[k][1]] < (1.0 + 0.5) * (minVal):  # 허용 오차 늘려서 에러 다시 구함

					where_black2.append(where_black[k])
			if len(where_black2) == 0:
				for k in range(len(where_black)):  # foam data 일정이상 하얀부분인 인덱스 중
					where_black2.append(where_black[k])

			c = np.random.randint(len(where_black2))
			y = where_black2[c][0]
			x = where_black2[c][1]
		else:  # 검은부분의 해당위치의 err가 허용오차 값 이하인 블록이 존재할때 -> 랜덤하게
			c = np.random.randint(len(where_black2))
			y = where_black2[c][0]
			x = where_black2[c][1]


	# count = []
	# for i in range(len(y)):
	# 	tmp = 0
	# 	for j in range(y[i], y[i] + blocksize + 1):
	# 		for k in range(x[i], x[i] + blocksize + 1):
	# 			if (mask[(blkIndexI):(blkIndexI + blocksize), (blkIndexJ):(blkIndexJ + blocksize)]==1).any():  # 마스크 안에 들어갈 경우
	# 				if (texture[j, k] >= 0.7).all():
	# 					tmp += 1
	# 			else:
	# 				if (texture[j, k] < 0.3).all():
	# 					tmp += 1
	# 	count.append(tmp)
	#
	# c = count.index(max(count))

	# #######
	# count = []
	# for i in range(len(y)):
	# 	tmp = 0
	# 	for j in range(y[i], y[i] + blocksize + 1):
	# 		for k in range(x[i], x[i] + blocksize + 1):
	# 			if (texture[j, k] >= 0.5).all():
	# 				tmp += 1
	# 	count.append(tmp)
	#
	# if (mask[(blkIndexI):(blkIndexI + blocksize), (blkIndexJ):(blkIndexJ + blocksize)]==1).any():
	# 	c = count.index(max(count))
	# else:
	# 	c = count.index(min(count))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)

	# y, x = y[c], x[c]  # 허용오차 안의 해당 에러중 랜덤하게 뽑음

	# plt.imshow(texture[y:y+blocksize, x:x+blocksize])  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
	# plt.show()

	return texture[y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return


# tan 추가 합성
def t_findPatchVertical(refBlock, texture, blocksize, overlap, tolerance, mask, blkIdx, where_white, where_black):
	'''
	Find best vertical match from the texture
	사용: findPatchVertical(refBlock, image, blocksize, overlap, tolerance)
	'''
	H, W = texture.shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
								# openCV 경우 -> (rows, columns, channels) 튜플 보유
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf	# np.zeros : 0으로 채워진 array 생성 / [[W-blocksize 만큼]*H-blocksize만큼] 0으로된 2차원 배열 생성

	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# [0,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		rmsVal = ((texture[i:i+overlap, j:j+blocksize] - refBlock[-overlap:, :])**2).mean()	# (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균

		if rmsVal > 0:
			errMat[i, j] = rmsVal	# 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	#minVal = np.min(errMat)	# 에러범위 값 중 가장 작은 것

	y = 0
	x = 0

	if (mask[(blkIdx):(blkIdx + blocksize), :blocksize]==1).any():  # 마스크 안에 들어갈 경우

		minVal = 1000
		c = 0
		for ii in range(len(where_white)):
			if errMat[where_white[ii][0], where_white[ii][1]] < minVal:
				minVal = errMat[where_white[ii][0], where_white[ii][1]]
				c = ii

		where_white2 = []
		for k in range(len(where_white)):  # foam data 일정이상 하얀부분인 인덱스 중
			if errMat[where_white[k][0], where_white[k][1]] < ((1.0 + tolerance) * (minVal)):  # err가 허용오차 값 이하일때의 인덱스 따로 빼둠
				where_white2.append(where_white[k])
		if len(where_white2) == 0:  # 하얀부분의 해당위치의 err가 허용오차 값 이하인 블록이 존재하지 않을 때
			for k in range(len(where_white)):  # foam data 일정이상 하얀부분인 인덱스 중
				if errMat[where_white[k][0], where_white[k][1]] < (1.0 + 0.3) * (minVal):  # 허용 오차 늘려서 에러 다시 구함
					where_white2.append(where_white[k])
			if len(where_white2)==0:
				for k in range(len(where_white)):  # foam data 일정이상 하얀부분인 인덱스 중
					where_white2.append(where_white[k])

			c = np.random.randint(len(where_white2))
			y = where_white2[c][0]
			x = where_white2[c][1]

			# minerr = 1000
			# c = 0
			# for ii in range(len(where_white)):
			# 	if errMat[where_white[ii][0], where_white[ii][1]] < minerr:
			# 		minerr = errMat[where_white[ii][0], where_white[ii][1]]
			# 		c = ii
			# print("final c : {}".format(c))
			# y = where_white[c][0]
			# x = where_white[c][1]
		else:	# 하얀부분의 해당위치의 err가 허용오차 값 이하인 블록이 존재할 때  -> 랜덤하게
			c = np.random.randint(len(where_white2))
			y = where_white2[c][0]
			x = where_white2[c][1]
	else:  # 마스크 밖 부분

		minVal = 1000
		c = 0
		for ii in range(len(where_black)):
			if errMat[where_black[ii][0], where_black[ii][1]] < minVal:
				minVal = errMat[where_black[ii][0], where_black[ii][1]]
				c = ii

		where_black2 = []
		for k in range(len(where_black)):  # foam data 일정이상 검은부분인 인덱스 중
			if errMat[where_black[k][0], where_black[k][1]] < (1.0 + tolerance) * (minVal):  # err가 허용오차 값 이하일때의 인덱스 따로 빼둠
				where_black2.append(where_black[k])
		if len(where_black2) == 0:  # 검은부분의 해당위치의 err가 허용오차 값 이하인 블록이 존재하지 않을 때
			for k in range(len(where_black)):  # foam data 일정이상 하얀부분인 인덱스 중
				if errMat[where_black[k][0], where_black[k][1]] < (1.0 + 0.5) * (minVal):  # 허용 오차 늘려서 에러 다시 구함
					where_black2.append(where_black[k])
			if len(where_black2) == 0:
				for k in range(len(where_black)):  # foam data 일정이상 검은부분인 인덱스 중
					where_black2.append(where_black[k])

			c = np.random.randint(len(where_black2))
			y = where_black2[c][0]
			x = where_black2[c][1]

			# minerr = 100
			# c = 0
			# for ii in range(len(where_black)):
			# 	if errMat[where_black[ii][0], where_black[ii][1]] < minerr:
			# 		minerr = errMat[where_black[ii][0], where_black[ii][1]]
			# 		c = ii
			#
			# y = where_black[c][0]
			# x = where_black[c][1]
		else:	# 검은부분의 해당위치의 err가 허용오차 값 이하인 블록이 존재할때 -> 랜덤하게
			c = np.random.randint(len(where_black2))
			y = where_black2[c][0]
			x = where_black2[c][1]

	# y, x = np.where(errMat < (1.0 + tolerance)*(minVal))	# np.where: 조건에 맞는 위치 인덱스 찾기 / 해당 허용오차보다 작은 E고름
	# 														# y : [뽑힌 원소 각각 행 어디인지]
	# 														# x : [뽑힌 원소 각각 열 어디인지] - (y,x) 둘이 이어서 위치 찾기
	#
	# count = []
	# for i in range(len(y)):
	# 	tmp = 0
	# 	for j in range(y[i], y[i] + blocksize + 1):
	# 		for k in range(x[i], x[i] + blocksize + 1):
	# 			if (mask[(blkIdx):(blkIdx + blocksize), :blocksize]==1).any():  # 마스크 안에 들어갈 경우
	# 				if (texture[j, k] >= 0.7).all():
	# 					tmp += 1
	# 			else:
	# 				if (texture[j, k] < 0.3).all():
	# 					tmp += 1
	# 	count.append(tmp)
	#
	# c = count.index(max(count))

	# ######
	# count = []
	# for i in range(len(y)):
	# 	tmp = 0
	# 	for j in range(y[i], y[i] + blocksize + 1):
	# 		for k in range(x[i], x[i] + blocksize + 1):
	# 			if (texture[j, k] >= 0.5).all():
	# 				tmp += 1
	# 	count.append(tmp)
	#
	# if (mask[(blkIdx):(blkIdx + blocksize), :blocksize]==1).any():
	# 	c = count.index(max(count))
	# 	print("11")
	# else:
	# 	c = count.index(min(count))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
	# 	print("22")

	# y, x = y[c], x[c]  # 허용오차 안의 해당 에러중 랜덤하게 뽑음

	# plt.imshow(texture[y:y+blocksize, x:x+blocksize])  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
	# plt.show()
	return texture[y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return

############################


#추가####################
# foam data 에 대한 합성
def foam_generateTextureMap(image, blocksize, overlap, outH, outW, tolerance):	# main.py에서 사용되는 메인. tolerance : 허용요차
	# 사용: generateTextureMap(image, block_size, overlap, outH, outW, args.tolerance)
	# ceil() : 소수점 자리의 숫자를 무조건 올리는 함수
	nH = int(ceil((outH - blocksize)*1.0/(blocksize - overlap)))	# 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?
	nW = int(ceil((outW - blocksize)*1.0/(blocksize - overlap)))	# 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?

	textureMap = np.zeros(((blocksize + nH*(blocksize - overlap)), (blocksize + nW*(blocksize - overlap)), image.shape[2]))
	# [(H기준 : nH(들어가는 블록개수) * (오버랩 뺀 블록실제사이즈) + 마지막에 오버랩 안되므로 블록 하나 더 사이즈) , (W기준 동일) , 색상] => 0으로 초기화
	# Starting index and block
	H, W = image.shape[:2]

	# # 첫 블록 유의미하게 랜덤값 #################
	# while True:
	# 	randH = np.random.randint(H - blocksize)	# 블록사이즈 한줄 뺀 값에서 랜덤한 값
	# 	randW = np.random.randint(W - blocksize)	# 블록사이즈 한줄 뺀 값에서 랜덤한 값
	#
	# 	count_black=0
	# 	a=0
	# 	for i in range(randH,randH + blocksize+1):
	# 		for j in range(randW,randW + blocksize+1):
	# 			a+=1
	# 			if (image[i,j]==[0,0,0]).all():
	# 				count_black += 1
	# 	if count_black<(blocksize*blocksize*(1/2)):
	# 		break
	# ###########################


	# tan 직선에 대한 mask ############
	a, b = textureMap.shape[:2]	# a = h, b = w
	c, d = H//2, W//2
	print("textruemap h: {}, w: {}".format(a, b))
	tan_mask = np.zeros((a,b))

	angle = 130	# 주어진 각도 - 회전된 직선 영역을 위하여
	slope = 0	# 회전된 직선영역의 기울기
	is_90 = False	# flag : 90도인가, 90도일경우에만 직선의 방정식 x= a 꼴이기 때문

	if angle%90==0:
		if angle%180==0:	# 180도 일 경우 y = y1 꼴
			slope = 0
		else:	# 90도 일 경우 x = x1 꼴
			is_90 = True
	else:	# 90, 180 도 배수 제외한 나머지 일 경우 y = ax + b
		slope = math.tan(math.radians(angle))

	flag1 = 0
	flagi = 0
	flagj = 0

	tmpj = 0
	for y in range(a):
		t = 0
		for x in range(b):
			if is_90 == True:
				tan_line = d
			else:
				tan_line = (a-1) - (math.ceil(slope * (x - d)) + c)	# 정해진 각도를 기울기로 갖는 이미지 상 직선

			if tan_line-30<=y and y<=tan_line+30:	# 기울어진 직선에서 얼만큼 두께를 줄 것인지
				tan_mask[y,x] = 1
				tmpj = x
				t+=1
				textureMap[y,x]=(255,0,0)

				if flag1 == 0:
					flag1 = 1
					flagi = y
					flagj = x
		# if t==0:	# for 문  y -> x 순으로 확인할 때 tan_line 이 짝수가 나오는 식이면 y 가 홀수일때 조건 만족하는 x를 찾을 수 없기 때문에 이전값을 저장했다가 그대로 씀
		# 	tan_mask[y,tmpj] = 1
		# 	textureMap[y, tmpj] = (255, 0, 0)

	plt.imshow(textureMap)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
	plt.show()
	# textureMap[flagi:flagi+blocksize, flagj:flagj+blocksize, :] = startBlock  # 0으로 초기화된 맵에서 첫번째 블록에 랜덤하게 가져온 블록 대입함

	# 이제 만들어갈 texturemap 의 첫 블록 - 랜덤하게 끼워넣음
	randH = np.random.randint(H - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값
	randW = np.random.randint(W - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값

	startBlock = image[randH:randH + blocksize, randW:randW + blocksize]  # 랜덤한 위치에서 시작하는 블록 사이즈만큼 잘라서 가져옴
	textureMap[:blocksize, :blocksize, :] = startBlock  # 0으로 초기화된 맵에서 첫번째 블록에 랜덤하게 가져온 블록 대입함
	######################

	where_white = []
	where_black = []

	for i in range(0, H-blocksize, blocksize):
		for j in range(0,W-blocksize, blocksize):
			count_black = 0
			for si in range(i, i + blocksize):
				for sj in range(j, j + blocksize):
					if (image[si, sj] == [0, 0, 0]).all():
						count_black += 1
			if count_black < (blocksize * blocksize * (1 / 2)):
				where_white.append([i,j])
			elif count_black > (blocksize * blocksize * (1 / 2)):
				where_black.append([i,j])

	################################

	# Fill the first row : 행(아래 위)
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[1]-overlap, (blocksize-overlap))):	# enumerate() : 인덱스와 원소 차례로 반환
		# 오버랩 부분 제외 블록 부분부터 ~ 오버랩 제외 열들까지 , 오버랩 제외한 블록사이즈만큼 옆으로 이동 (오른 -> 왼)

		# Find horizontal error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		# blkIdx = 블록에서 오버랩 되는 부분 시작점 인덱스
		refBlock = textureMap[:blocksize, (blkIdx-blocksize+overlap):(blkIdx+overlap)]	#texturemap 의 한줄제외 모든 행에 대하여 열단위로 블록 한 칸만큼 계속 이동하면서 대입
		patchBlock = t_findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance, tan_mask, blkIdx, where_white, where_black)	# 미리 만든 패치 찾는 함수
		minCutPatch = getMinCutPatchHorizontal(refBlock, patchBlock, blocksize, overlap)	# 미리 만든 최소 경로 찾는 함수
		textureMap[:blocksize, (blkIdx):(blkIdx+blocksize)] = minCutPatch	# 오버랩부분 경계선 최소경로로 자름
	print("{} out of {} rows complete...".format(1, nH+1))


	### Fill the first column 열 (오른 왼쪽)
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[0]-overlap, (blocksize-overlap))):	# # enumerate() : 인덱스와 원소 차례로 반환
		# 오버랩 부분 제외 블록 부분부터 ~ 오버랩 제외 행들까지 , 오버랩 제외한 블록사이즈만큼 옆으로 이동 (위 -> 아래)

		# Find vertical error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		# blkIdx = 블록에서 오버랩 되는 부분 시작점 인덱스
		refBlock = textureMap[(blkIdx-blocksize+overlap):(blkIdx+overlap), :blocksize]	#texturemap 의 한줄제외 모든 열에 대하여 행단위로 블록 한 칸만큼 계속 이동하면서 대입
		patchBlock = t_findPatchVertical(refBlock, image, blocksize, overlap, tolerance, tan_mask, blkIdx, where_white, where_black)	# 미리 만든 패치 찾는 함수
		minCutPatch = getMinCutPatchVertical(refBlock, patchBlock, blocksize, overlap)	# 미리 만든 최소 경로 찾는 함수
		textureMap[(blkIdx):(blkIdx+blocksize), :blocksize] = minCutPatch	# 오버랩부분 경계선 최소경로로 자름

	### Fill in the other rows and columns
	for i in range(1, nH+1):
		for j in range(1, nW+1):
			# Choose the starting index for the texture placement
			blkIndexI = i*(blocksize-overlap)
			blkIndexJ = j*(blocksize-overlap)
			# Find the left and top block, and the min errors independently
			refBlockLeft = textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ-blocksize+overlap):(blkIndexJ+overlap)]
			refBlockTop  = textureMap[(blkIndexI-blocksize+overlap):(blkIndexI+overlap), (blkIndexJ):(blkIndexJ+blocksize)]

			patchBlock = t_findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance, tan_mask, blkIndexI, blkIndexJ, where_white, where_black)
			minCutPatch = getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap)

			textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ):(blkIndexJ+blocksize)] = minCutPatch

			# refBlockLeft = 0.5
			# textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ-blocksize+overlap):(blkIndexJ+overlap)] = refBlockLeft
			# textureMap[(blkIndexI-blocksize+overlap):(blkIndexI+overlap), (blkIndexJ):(blkIndexJ+blocksize)] = [0.5, 0.6, 0.7]
			# break
		print("{} out of {} rows complete...".format(i+1, nH+1))
		# break

	plt.imshow(textureMap)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
	plt.show()

	return textureMap
#####################

#추가###################
# 8회전 합성 부분
def r_generateTextureMap(image, blocksize, overlap, y, x, tolerance, mask):	# 회전이미지에서 검은부분 합성으로 채우기
	# 사용: generateTextureMap(image, block_size, overlap, outH, outW, args.tolerance)
	# ceil() : 소수점 자리의 숫자를 무조건 올리는 함수
	nH = int(ceil((y - blocksize) * 1.0 / (blocksize - overlap)))  # 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?
	nW = int(ceil((x - blocksize) * 1.0 / (blocksize - overlap)))  # 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?
	textureMap = np.zeros(((blocksize + nH * (blocksize - overlap)), (blocksize + nW * (blocksize - overlap)), image.shape[2]))
	# [(H기준 : nH(들어가는 블록개수) * (오버랩 뺀 블록실제사이즈) + 마지막에 오버랩 안되므로 블록 하나 더 사이즈) , (W기준 동일) , 색상] => 0으로 초기화
	# Starting index and block
	H, W = image.shape[:2]

	while (True):  # do-while 문
		randH = np.random.randint(H - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값
		randW = np.random.randint(W - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값

		if (mask[randH:randH + blocksize, randW:randW + blocksize] == 1).all():  # 로테이션 이미지 존재한는 부분일 때의 random 값 뽑아내기
			break

	startBlock = image[randH:randH + blocksize, randW:randW + blocksize]  # 랜덤한 위치에서 시작하는 블록 사이즈만큼 잘라서 가져옴
	textureMap[:blocksize, :blocksize, :] = startBlock  # 0으로 초기화된 맵에서 첫번째 블록에 랜덤하게 가져온 블록 대입함

	# Fill the first row : 행(아래 위)
	for i, blkIdx in enumerate(range((blocksize - overlap), textureMap.shape[1] - overlap, (blocksize - overlap))):  # enumerate() : 인덱스와 원소 차례로 반환
		# 오버랩 부분 제외 블록 부분부터 ~ 오버랩 제외 열들까지 , 오버랩 제외한 블록사이즈만큼 옆으로 이동 (오른 -> 왼)

		# Find horizontal error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		# blkIdx = 블록에서 오버랩 되는 부분 시작점 인덱스
		refBlock = textureMap[:blocksize, (blkIdx - blocksize + overlap):(blkIdx + overlap)]  # texturemap 의 한줄제외 모든 행에 대하여 열단위로 블록 한 칸만큼 계속 이동하면서 대입
		patchBlock = r_findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance, mask)  # 미리 만든 패치 찾는 함수
		minCutPatch = getMinCutPatchHorizontal(refBlock, patchBlock, blocksize, overlap)  # 미리 만든 최소 경로 찾는 함수
		textureMap[:blocksize, (blkIdx):(blkIdx + blocksize)] = minCutPatch  # 오버랩부분 경계선 최소경로로 자름
	print("{} out of {} rows complete...".format(1, nH + 1))

	### Fill the first column 열 (오른 왼쪽)
	for i, blkIdx in enumerate(range((blocksize - overlap), textureMap.shape[0] - overlap, (blocksize - overlap))):  # # enumerate() : 인덱스와 원소 차례로 반환
		# 오버랩 부분 제외 블록 부분부터 ~ 오버랩 제외 행들까지 , 오버랩 제외한 블록사이즈만큼 옆으로 이동 (위 -> 아래)

		# Find vertical error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		# blkIdx = 블록에서 오버랩 되는 부분 시작점 인덱스
		refBlock = textureMap[(blkIdx - blocksize + overlap):(blkIdx + overlap), :blocksize]  # texturemap 의 한줄제외 모든 열에 대하여 행단위로 블록 한 칸만큼 계속 이동하면서 대입
		patchBlock = r_findPatchVertical(refBlock, image, blocksize, overlap, tolerance, mask)  # 미리 만든 패치 찾는 함수
		minCutPatch = getMinCutPatchVertical(refBlock, patchBlock, blocksize, overlap)  # 미리 만든 최소 경로 찾는 함수
		textureMap[(blkIdx):(blkIdx + blocksize), :blocksize] = minCutPatch  # 오버랩부분 경계선 최소경로로 자름

	### Fill in the other rows and columns
	for i in range(1, nH + 1):
		for j in range(1, nW + 1):
			# Choose the starting index for the texture placement
			blkIndexI = i * (blocksize - overlap)
			blkIndexJ = j * (blocksize - overlap)
			# Find the left and top block, and the min errors independently
			refBlockLeft = textureMap[(blkIndexI):(blkIndexI + blocksize), (blkIndexJ - blocksize + overlap):(blkIndexJ + overlap)]
			refBlockTop = textureMap[(blkIndexI - blocksize + overlap):(blkIndexI + overlap), (blkIndexJ):(blkIndexJ + blocksize)]

			patchBlock = r_findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance, mask)
			minCutPatch = getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap)

			textureMap[(blkIndexI):(blkIndexI + blocksize), (blkIndexJ):(blkIndexJ + blocksize)] = minCutPatch

		# refBlockLeft = 0.5
		# textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ-blocksize+overlap):(blkIndexJ+overlap)] = refBlockLeft
		# textureMap[(blkIndexI-blocksize+overlap):(blkIndexI+overlap), (blkIndexJ):(blkIndexJ+blocksize)] = [0.5, 0.6, 0.7]
		# break
		print("{} out of {} rows complete...".format(i + 1, nH + 1))
	# break

	return textureMap

#######################

# 전처리 : 1. 예제 이미지의 전처리
def Pre_RotateExImg(image, exImg, blocksize, overlap, outH, outW, tolerance):  # 방향성 더해주기 위한 내가만든 함수
	# 사용: generateTextureMap(image, block_size, overlap, outH, outW, args.tolerance)
	# ceil() : 소수점 자리의 숫자를 무조건 올리는 함수
	nH = int(ceil((outH - blocksize) * 1.0 / (blocksize - overlap)))  # 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?
	nW = int(ceil((outW - blocksize) * 1.0 / (blocksize - overlap)))  # 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?

	####try################################################
	# 이미지의 크기를 잡고 이미지의 중심을 계산합니다.
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)
	is_toroidal = []

	img8 = []

	for i in range(1, 9):
		imax = 8
		r_seta = i / imax * 360

		# 90도 배수이면 toroidal = 1
		if (r_seta % 90 == 0):
			is_toroidal.append(1)
		else:
			is_toroidal.append(0)

		# 이미지의 중심을 중심으로 이미지를 r_seta도 회전합니다.
		M = cv2.getRotationMatrix2D((cX, cY), r_seta, 1.0)	# cv2.getRotationMatrix2D(회전중심좌표(x,y 튜플), 회전각도, 스케일)
		rotated_seta = cv2.warpAffine(image, M, (h, w))	# cv2.warpAffine(src 원본이미지, M 아핀 맵 행렬, dsize 출력 이미지 크기) : 회전 변환을 계산

		# 준비된 합성된 예제 이미지도 같이 회전시킴
		#rotated_exImg = cv2.warpAffine(exImg, M, (w, h))

		# ex = generateTextureMap(rotated_seta, blocksize, overlap, h, w, tolerance)
		# plt.imshow(ex)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
		# plt.show()

		# # 0,0,0 : 검은 부분일 경우
		# for h_rimg in range(rotated_seta.shape[0]):
		# 	for w_rimg in range(rotated_seta.shape[1]):
		# 		if np.all(rotated_seta[h_rimg, w_rimg] == 0):
		# 			if is_toroidal[i - 1] == 1:  # toroidal : 타일 로 대체
		# 				rotated_seta[h_rimg, w_rimg] = image[h_rimg, w_rimg]
		# 			elif is_toroidal[i - 1] == 0:  # non-torodial : 미러링 으로 대체
		# 				rotated_seta[h_rimg, w_rimg] = image[h - 1 - h_rimg, w - 1 - w_rimg]
		#
		# plt.imshow(rotated_seta)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
		# plt.show()

		# # 검은 부분 : tan 로 계산 경우
		# half_seta = (r_seta%90)/2
		# tan_seta = math.tan(half_seta)
		# L = w/2
		# print("half:{} , tan: {}, L:{}".format(half_seta,tan_seta,L))
		# line_1 = int((tan_seta+1)*L)
		# line_2 = int((((1-tan_seta)/(1+tan_seta))-tan_seta)*L)
		# line_3 = int((1-((1-tan_seta)/(1+tan_seta)))*L)
		# line_4 = int(L*(1-math.tan(r_seta)*(1/2)*(tan_seta+1)))
		#
		# print("1:{} , 2: {}, 4:{}".format(line_1,line_2,line_4))
		#
		# tmp = line_1
		# for row in range(20):
		# 	for col in range(20):
		# 		rotated_seta[row,col] = [255,0,0]
		# 	tmp -= 1

		# 검은 부분 : sin cos  로 계산 경우
		L = w
		sin_seta = math.sin(r_seta%90)
		cos_seta = math.cos(r_seta%90)
		t = L/(sin_seta+cos_seta+1)

		line_1 = ceil(t * sin_seta)	# 넉넉하게 라인 길이를 잡아줘야 하므로 올림으로 하였다.
		line_2 = ceil(t)
		line_3 = ceil(t * cos_seta)
		print("sin:{} , cos: {}, t:{}".format(sin_seta,cos_seta,t))
		print("1:{} , 2: {}, 3:{}".format(line_1, line_2, line_3))
		print("h: {}, W: {}".format(h,w))

		# #기존 확인용###############
		# fill_black_img = rotated_seta.copy()
		# # 왼쪽 위 부분
		# tmp = line_1
		# for y in range(line_1):	# 원래대로라면 line_3 이 들어가야 하지만 검은 삼각형이 w,h가 같지않으므로 위에서부터 1칸씩 빼면서 내려가면 깔끔하게 삼각형이 안채워져서 원본이 정사각형이라는 가정 하(이것도 상관없는것같긴한데..)에 깔끔한 검은삼각형을 채우기 위하여 w,h가 같다고 가정하고 채워주기 위해 w=h 로 하였다..
		# 	for x in range(tmp):
		# 		fill_black_img[y, x] = [1, 0, 0]
		# 	tmp -= 1
		# # 왼쪽 아래 부분
		# tmp = 1
		# for y in range(L-line_1,L):
		# 	for x in range(tmp):
		# 		fill_black_img[y, x] = [1, 0, 0]
		# 	tmp += 1
		# # 오른쪽 위 부분
		# tmp = line_1
		# for y in range(line_1):
		# 	for x in range(L-tmp,L):
		# 		fill_black_img[y, x] = [1, 0, 0]
		# 	tmp -= 1
		# # 오른쪽 아래 부분
		# tmp = 0
		# for y in range(L-line_1,L):  # 원래대로라면 line_3 이 들어가야 하지만 검은 삼각형이 w,h가 같지않으므로 위에서부터 1칸씩 빼면서 내려가면 깔끔하게 삼각형이 안채워져서 원본이 정사각형이라는 가정 하(이것도 상관없는것같긴한데..)에 깔끔한 검은삼각형을 채우기 위하여 w,h가 같다고 가정하고 채워주기 위해 w=h 로 하였다..
		# 	for x in range(L-tmp,L):
		# 		fill_black_img[y, x] = [1, 0, 0]
		# 	tmp += 1
		# pre_img = cv2.addWeighted(rotated_seta, 0.5, fill_black_img, 0.5, 0)
		# #############################

## 수정1
		mask_black = np.ones((h, w, 3))
		black_h = line_1	# 검은 삼각형 높이부분 : tsin@
		black_w	= line_1	# 검은 삼각형 밑변부분 : tcos@

		# rotation -> 검은 삼각형 부분 => 마스크 만들기
		# 왼쪽 위 부분
		tmp = black_h
		for y in range(black_w):	# 원래대로라면 line_3 이 들어가야 하지만 검은 삼각형이 w,h가 같지않으므로 위에서부터 1칸씩 빼면서 내려가면 깔끔하게 삼각형이 안채워져서 원본이 정사각형이라는 가정 하(이것도 상관없는것같긴한데..)에 깔끔한 검은삼각형을 채우기 위하여 w,h가 같다고 가정하고 채워주기 위해 w=h 로 하였다..
			for x in range(tmp):
				mask_black[y, x] = 0
			tmp -= 1
		# 왼쪽 아래 부분
		tmp = 1
		for y in range(L-black_h,L):
			for x in range(tmp):
				mask_black[y, x] = 0
			tmp += 1
		# 오른쪽 위 부분
		tmp = black_w
		for y in range(black_h):
			for x in range(L-tmp,L):
				mask_black[y, x] = 0
			tmp -= 1
		# 오른쪽 아래 부분
		tmp = 0
		for y in range(L-black_h,L):  # 원래대로라면 line_3 이 들어가야 하지만 검은 삼각형이 w,h가 같지않으므로 위에서부터 1칸씩 빼면서 내려가면 깔끔하게 삼각형이 안채워져서 원본이 정사각형이라는 가정 하(이것도 상관없는것같긴한데..)에 깔끔한 검은삼각형을 채우기 위하여 w,h가 같다고 가정하고 채워주기 위해 w=h 로 하였다..
			for x in range(L-tmp,L):
				mask_black[y, x] = 0
			tmp += 1

		# rotation -> 검은 삼각형 부분 => 합성 #########
		r_texture_black = r_generateTextureMap(rotated_seta, blocksize, overlap, h, w, tolerance, mask_black)	# 방향성 고려해서 새로 합성한 후보이미지

		# 어차피 회전 예제 이미지의 방향값을 가져오는 것이 목적이므로 더 자연스러운 새로만든 텍스쳐를 사용한다.
		r_texture_black1 = r_texture_black[:h, :w, :]	# r_generateTextureMap () 함수 시 블록 사이즈에 나눠떨어지게 크기가 생성되므로 h,w 라도 좀 더 크게 잡힌다. 따라서 크기가 달라 아래에서 연산이 안되므로 조절해준다.
		r_texture = rotated_seta * mask_black + r_texture_black1 * (1-mask_black)	# 기존 이미지 + 방향성 합성 이미지 검은부분용

		# plt.imshow(r_texture_black)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
		# plt.show()

		img8.append(r_texture_black)

		# # Save
		# pre_img = (255 * r_texture_black).astype(np.uint8)  # 최종 결과 텍스쳐 맵 -> 0~1, RGB 형태 => 원래대로로 돌림 (0~155 , BGR형태 , unit8)
		# pre_img = cv2.cvtColor(pre_img, cv2.COLOR_RGB2BGR)
		#
		# cv2.imwrite("8img_" + str(i) + ".png", pre_img)
		#
		# pre_img1 = (255 * r_texture).astype(np.uint8)  # 최종 결과 텍스쳐 맵 -> 0~1, RGB 형태 => 원래대로로 돌림 (0~155 , BGR형태 , unit8)
		# pre_img1 = cv2.cvtColor(pre_img1, cv2.COLOR_RGB2BGR)
		#
		# cv2.imwrite("10img_" + str(i) + ".png", pre_img1)

	return img8

# 전처리 : 2. 합성 대상 이미지 S의 재정의
def Pre_AddRotateIndex(img8):
	for i in range(len(img8)):
		img = img8[i]
		h = img.shape[0]
		w = img.shape[1]

		addArray = np.full((h,w,1), i)
		new = np.concatenate([img, addArray], axis=2)
	print("new : {}".format(new))
	return new

def Pre_FindNeighbor(img8,ref,size):
	print("함수 안+++++")
	NEi = ref
	half = int(size//2)
	print("half={}".format(half))
	for i in range(len(img8)):
		print("111")
		img = img8[i]
		h = img.shape[0]
		w = img.shape[1]
		tmp_err=100
		tmp_p=[]
		tmp_j=0

		for y in range(half-1,h-half+1):
			print("222")
			print("imageshape:{}".format(img.shape))
			print("{} w:{}".format(h,w))
			print("{} ~ {}".format(half-1, w-half+1))
			for x in range(half-1,w-half+1):
				print("333")
				# 예외처리 - 안해주면 통째로 [] 로 처리돼서 error 계산시 NaN 으로 나옴 (숫자 아니라는 뜻)
				NEj = img[y-(half-1):y+half+1,x-(half-1):x+half+1]


				print("{},{} 에서의 ".format(y, x))
				print("i:{} , j:{}".format(NEi.shape, NEj.shape))

				err = ((NEj[:,:] - NEi[:,:]) ** 2).mean()	# error 를 어떻게 구하는지에 대한 언급이 없어서 기존 error 구하는 공식 가져옴
				print("err: {}".format(err))
				if err<tmp_err:
					print("eee")
					tmp_err = err
					tmp_p = NEj
					tmp_j = i

	return tmp_p



def fin_findPatchHorizontal(refBlock, texture, blocksize, overlap, tolerance, img8):	# tolerance : 허용오차
	'''
	Find best horizontal match from the texture
	사용: findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance)
	'''
	H, W = texture.shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf	# np.zeros : 0으로 채워진 array 생성 / [[W-blocksize 만큼]*H-blocksize만큼] 0으로된 2차원 배열 생성 / inf 는 왜더하는지는 모르겠음???
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# openCV 경우 -> (rows, columns, channels) 튜플 보유,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		rmsVal = ((texture[i:i+blocksize, j:j+overlap] - refBlock[:, -overlap:])**2).mean()	# (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균
		if rmsVal > 0:
			errMat[i, j] = rmsVal	# 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	print("시작+++++")
	Pre_FindNeighbor(img8, refBlock, blocksize)

	minVal = np.min(errMat)	# 에러범위 값 중 가장 작은 것
	y, x = np.where(errMat < (1.0 + tolerance)*(minVal))	# np.where: 조건에 맞는 위치 인덱스 찾기 / 해당 허용오차보다 작은 E 고름
															# y : [뽑힌 원소 각각 행 어디인지]
															# x : [뽑힌 원소 각각 열 어디인지] - (y,x) 둘이 이어서 위치 찾기
	c = np.random.randint(len(y))	# random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
	y, x = y[c], x[c]	# 허용오차 안의 해당 에러 중 랜덤하게 뽑음

	# for tmpH in range(H):
	# 	for tmeW in range(W):

	return texture[y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return



def fin_findPatchBoth(refBlockLeft, refBlockTop, texture, blocksize, overlap, tolerance, img8):
	'''
	Find best horizontal and vertical match from the texture
	사용: findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance)
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



def fin_findPatchVertical(refBlock, texture, blocksize, overlap, tolerance, img8):
	'''
	Find best vertical match from the texture
	사용: findPatchVertical(refBlock, image, blocksize, overlap, tolerance)
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


def fin_generateTextureMap(image, blocksize, overlap, outH, outW, tolerance):	# main.py에서 사용되는 메인. tolerance : 허용요차
	# 사용: generateTextureMap(image, block_size, overlap, outH, outW, args.tolerance)
	# ceil() : 소수점 자리의 숫자를 무조건 올리는 함수
	nH = int(ceil((outH - blocksize)*1.0/(blocksize - overlap)))	# 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?
	nW = int(ceil((outW - blocksize)*1.0/(blocksize - overlap)))	# 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?

	textureMap = np.zeros(((blocksize + nH*(blocksize - overlap)), (blocksize + nW*(blocksize - overlap)), image.shape[2]))
	# [(H기준 : nH(들어가는 블록개수) * (오버랩 뺀 블록실제사이즈) + 마지막에 오버랩 안되므로 블록 하나 더 사이즈) , (W기준 동일) , 색상] => 0으로 초기화
	# Starting index and block
	H, W = image.shape[:2]
	pre_img8 = Pre_RotateExImg(image, image, blocksize, overlap, outH, outW, tolerance)
	#pre_img8 = Pre_AddRotateIndex(pre_img8)

	randH = np.random.randint(H - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값
	randW = np.random.randint(W - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값

	startBlock = image[randH:randH+blocksize, randW:randW+blocksize]	# 랜덤한 위치에서 시작하는 블록 사이즈만큼 잘라서 가져옴
	textureMap[:blocksize, :blocksize, :] = startBlock	# 0으로 초기화된 맵에서 첫번째 블록에 랜덤하게 가져온 블록 대입함

	# Fill the first row : 행(아래 위)
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[1]-overlap, (blocksize-overlap))):	# enumerate() : 인덱스와 원소 차례로 반환
		# 오버랩 부분 제외 블록 부분부터 ~ 오버랩 제외 열들까지 , 오버랩 제외한 블록사이즈만큼 옆으로 이동 (오른 -> 왼)

		# Find horizontal error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		# blkIdx = 블록에서 오버랩 되는 부분 시작점 인덱스
		refBlock = textureMap[:blocksize, (blkIdx-blocksize+overlap):(blkIdx+overlap)]	#texturemap 의 한줄제외 모든 행에 대하여 열단위로 블록 한 칸만큼 계속 이동하면서 대입
		patchBlock = fin_findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance, pre_img8)	# 미리 만든 패치 찾는 함수
		minCutPatch = getMinCutPatchHorizontal(refBlock, patchBlock, blocksize, overlap)	# 미리 만든 최소 경로 찾는 함수
		textureMap[:blocksize, (blkIdx):(blkIdx+blocksize)] = minCutPatch	# 오버랩부분 경계선 최소경로로 자름
	print("{} out of {} rows complete...".format(1, nH+1))


	### Fill the first column 열 (오른 왼쪽)
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[0]-overlap, (blocksize-overlap))):	# # enumerate() : 인덱스와 원소 차례로 반환
		# 오버랩 부분 제외 블록 부분부터 ~ 오버랩 제외 행들까지 , 오버랩 제외한 블록사이즈만큼 옆으로 이동 (위 -> 아래)

		# Find vertical error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		# blkIdx = 블록에서 오버랩 되는 부분 시작점 인덱스
		refBlock = textureMap[(blkIdx-blocksize+overlap):(blkIdx+overlap), :blocksize]	#texturemap 의 한줄제외 모든 열에 대하여 행단위로 블록 한 칸만큼 계속 이동하면서 대입
		patchBlock = fin_findPatchVertical(refBlock, image, blocksize, overlap, tolerance, pre_img8)	# 미리 만든 패치 찾는 함수
		minCutPatch = getMinCutPatchVertical(refBlock, patchBlock, blocksize, overlap)	# 미리 만든 최소 경로 찾는 함수
		textureMap[(blkIdx):(blkIdx+blocksize), :blocksize] = minCutPatch	# 오버랩부분 경계선 최소경로로 자름

	### Fill in the other rows and columns
	for i in range(1, nH+1):
		for j in range(1, nW+1):
			# Choose the starting index for the texture placement
			blkIndexI = i*(blocksize-overlap)
			blkIndexJ = j*(blocksize-overlap)
			# Find the left and top block, and the min errors independently
			refBlockLeft = textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ-blocksize+overlap):(blkIndexJ+overlap)]
			refBlockTop  = textureMap[(blkIndexI-blocksize+overlap):(blkIndexI+overlap), (blkIndexJ):(blkIndexJ+blocksize)]

			patchBlock = fin_findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance, pre_img8)
			minCutPatch = getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap)

			textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ):(blkIndexJ+blocksize)] = minCutPatch

			# refBlockLeft = 0.5
			# textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ-blocksize+overlap):(blkIndexJ+overlap)] = refBlockLeft
			# textureMap[(blkIndexI-blocksize+overlap):(blkIndexI+overlap), (blkIndexJ):(blkIndexJ+blocksize)] = [0.5, 0.6, 0.7]
			# break
		print("{} out of {} rows complete...".format(i+1, nH+1))
		# break

	return textureMap

# 여러 회전각도에 대하여 회전시키는 시도
def multi_RotateExImg(image, blocksize, overlap, outH, outW, tolerance):  # 방향성 더해주기 위한 내가만든 함수
	# 사용: generateTextureMap(image, block_size, overlap, outH, outW, args.tolerance)
	# ceil() : 소수점 자리의 숫자를 무조건 올리는 함수
	nH = int(ceil((outH - blocksize) * 1.0 / (blocksize - overlap)))  # 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?
	nW = int(ceil((outW - blocksize) * 1.0 / (blocksize - overlap)))  # 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?

	####try################################################
	# 이미지의 크기를 잡고 이미지의 중심을 계산합니다.
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)

	rImgs = []

	for i in range(0, 6, 5):
		imax = 360
		r_seta = i

		# 이미지의 중심을 중심으로 이미지를 r_seta도 회전합니다.
		M = cv2.getRotationMatrix2D((cX, cY), r_seta, 1.0)	# cv2.getRotationMatrix2D(회전중심좌표(x,y 튜플), 회전각도, 스케일)
		rotated_seta = cv2.warpAffine(image, M, (h, w))	# cv2.warpAffine(src 원본이미지, M 아핀 맵 행렬, dsize 출력 이미지 크기) : 회전 변환을 계산

		# 검은 부분 : sin cos  로 계산 경우
		X = w
		sin_seta = math.sin((r_seta + 45))
		if sin_seta<0:
			sin_seta = -sin_seta
		cos_seta = math.cos((r_seta + 45))
		if cos_seta<0:
			cos_seta = -cos_seta
		a = X / (cos_seta + 1 + sin_seta)

		line_1 = ceil(a * cos_seta) + 30	# 넉넉하게 라인 길이를 잡아줘야 하므로 올림으로 하였다.
		line_2 = ceil(a)
		line_3 = ceil(a * sin_seta) + 30
		print("sin:{} , cos: {}, x:{}".format(sin_seta,cos_seta,a))
		print("1:{} , 2: {}, 3:{}".format(line_1, line_2, line_3))
		print("h: {}, W: {}".format(h,w))

		rangeLine = [line_1,line_3]
		longline=w+100
		for line in rangeLine:
			if longline>=line:
				longline = line
		print("liongline:{}".format(longline))

		mask_black = np.ones((h, w, 3))
		#기존 확인용###############
		fill_black_img = rotated_seta.copy()
		# 왼쪽 위 부분
		tmp = longline
		for y in range(longline):	# 원래대로라면 line_3 이 들어가야 하지만 검은 삼각형이 w,h가 같지않으므로 위에서부터 1칸씩 빼면서 내려가면 깔끔하게 삼각형이 안채워져서 원본이 정사각형이라는 가정 하(이것도 상관없는것같긴한데..)에 깔끔한 검은삼각형을 채우기 위하여 w,h가 같다고 가정하고 채워주기 위해 w=h 로 하였다..
			for x in range(tmp):
				mask_black[y, x] = 0
				fill_black_img[y, x] = [1, 0, 0]
			tmp -= 1
		# 왼쪽 아래 부분
		tmp = 1
		for y in range(X-longline,X):
			for x in range(tmp):
				mask_black[y, x] = 0
				fill_black_img[y, x] = [1, 0, 0]
			tmp += 1
		# 오른쪽 위 부분
		tmp = longline
		for y in range(longline):
			for x in range(X-tmp,X):
				mask_black[y, x] = 0
				fill_black_img[y, x] = [1, 0, 0]
			tmp -= 1
		# 오른쪽 아래 부분
		tmp = 0
		for y in range(X-longline,X):  # 원래대로라면 line_3 이 들어가야 하지만 검은 삼각형이 w,h가 같지않으므로 위에서부터 1칸씩 빼면서 내려가면 깔끔하게 삼각형이 안채워져서 원본이 정사각형이라는 가정 하(이것도 상관없는것같긴한데..)에 깔끔한 검은삼각형을 채우기 위하여 w,h가 같다고 가정하고 채워주기 위해 w=h 로 하였다..
			for x in range(X-tmp,X):
				mask_black[y, x] = 0
				fill_black_img[y, x] = [1, 0, 0]
			tmp += 1
		pre_img = cv2.addWeighted(rotated_seta, 0.5, fill_black_img, 0.5, 0)
		rImgs.append(pre_img)

		#rImgs.append(fill_black_img)
		# plt.imshow(fill_black_img)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
		# plt.show()
		#############################

# ## 수정1
# 		mask_black = np.ones((h, w, 3))
# 		black_h = longline	# 검은 삼각형 높이부분 : tsin@
# 		black_w	= longline	# 검은 삼각형 밑변부분 : tcos@
#
# 		# rotation -> 검은 삼각형 부분 => 마스크 만들기
# 		# 왼쪽 위 부분
# 		tmp = black_h
# 		for y in range(black_w):	# 원래대로라면 line_3 이 들어가야 하지만 검은 삼각형이 w,h가 같지않으므로 위에서부터 1칸씩 빼면서 내려가면 깔끔하게 삼각형이 안채워져서 원본이 정사각형이라는 가정 하(이것도 상관없는것같긴한데..)에 깔끔한 검은삼각형을 채우기 위하여 w,h가 같다고 가정하고 채워주기 위해 w=h 로 하였다..
# 			for x in range(tmp):
# 				mask_black[y, x] = 0
# 			tmp -= 1
# 		# 왼쪽 아래 부분
# 		tmp = 1
# 		for y in range(X-black_h,X):
# 			for x in range(tmp):
# 				mask_black[y, x] = 0
# 			tmp += 1
# 		# 오른쪽 위 부분
# 		tmp = black_w
# 		for y in range(black_h):
# 			for x in range(X-tmp,X):
# 				mask_black[y, x] = 0
# 			tmp -= 1
# 		# 오른쪽 아래 부분
# 		tmp = 0
# 		for y in range(X-black_h,X):  # 원래대로라면 line_3 이 들어가야 하지만 검은 삼각형이 w,h가 같지않으므로 위에서부터 1칸씩 빼면서 내려가면 깔끔하게 삼각형이 안채워져서 원본이 정사각형이라는 가정 하(이것도 상관없는것같긴한데..)에 깔끔한 검은삼각형을 채우기 위하여 w,h가 같다고 가정하고 채워주기 위해 w=h 로 하였다..
# 			for x in range(X-tmp,X):
# 				mask_black[y, x] = 0
# 			tmp += 1

		# rotation -> 검은 삼각형 부분 => 합성 #########
		r_texture_black = r_generateTextureMap(rotated_seta, blocksize, overlap, h, w, tolerance, mask_black)	# 방향성 고려해서 새로 합성한 후보이미지

		# 어차피 회전 예제 이미지의 방향값을 가져오는 것이 목적이므로 더 자연스러운 새로만든 텍스쳐를 사용한다.
		r_texture_black1 = r_texture_black[:h, :w, :]	# r_generateTextureMap () 함수 시 블록 사이즈에 나눠떨어지게 크기가 생성되므로 h,w 라도 좀 더 크게 잡힌다. 따라서 크기가 달라 아래에서 연산이 안되므로 조절해준다.
		r_texture = rotated_seta * mask_black + r_texture_black1 * (1-mask_black)	# 기존 이미지 + 방향성 합성 이미지 검은부분용

		# plt.imshow(r_texture_black)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
		# plt.show()

		rImgs.append(r_texture_black)

		# # Save
		# pre_img = (255 * r_texture_black).astype(np.uint8)  # 최종 결과 텍스쳐 맵 -> 0~1, RGB 형태 => 원래대로로 돌림 (0~155 , BGR형태 , unit8)
		# pre_img = cv2.cvtColor(pre_img, cv2.COLOR_RGB2BGR)
		#
		# cv2.imwrite("8img_" + str(i) + ".png", pre_img)
		#
		# pre_img1 = (255 * r_texture).astype(np.uint8)  # 최종 결과 텍스쳐 맵 -> 0~1, RGB 형태 => 원래대로로 돌림 (0~155 , BGR형태 , unit8)
		# pre_img1 = cv2.cvtColor(pre_img1, cv2.COLOR_RGB2BGR)
		#
		# cv2.imwrite("10img_" + str(i) + ".png", pre_img1)

#### gif 만드는 부분
	gif_config = {
		'loop': 1,  ## 0으로 세팅하면 무한 반복, 3으로 설정하면 3번 반복
		'duration': 0.5  ## 다음 화면으로 넘어가는 시간
	}

	## gif로 만들 이미지를 리스트로 만들어 줌
	images = rImgs

	## mimwrite 대신 mimsave로도 가능
	imageio.mimwrite(os.path.join(os.getcwd(), 'result_4090.gif'),  ## 저장 경로
					 images,  ## 이미지 리스트
					 format='gif',  ## 저장 포맷
					 **gif_config  ## 부가 요소
					 )

	return rImgs