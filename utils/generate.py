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

ANGEL_NUM = 8

inf = float('inf')	# 그 자체로 ∞를 의미
ErrorCombinationFunc = np.add	# ?? np.add: array 요소 단위로 덧셈 계산..

#######################
# ++original++
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
#####################################################
#####################################################


#추가#########################

# 전처리 : 회전 이미지 생성
def Make_RotateExImg(image, blocksize, overlap, tolerance):  # 방향성 더해주기 위한 내가만든 함수
	print(">>Pre_roatateExImg")
	# 사용: generateTextureMap(image, block_size, overlap, outH, outW, args.tolerance)

	####try################################################
	# 이미지의 크기를 잡고 이미지의 중심을 계산합니다.
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)
	# is_toroidal = []

	img8 = []

	for i in range(1,ANGEL_NUM+1):
		imax = ANGEL_NUM
		r_seta = i / imax * 360

		# # 90도 배수이면 toroidal = 1
		# if (r_seta % 90 == 0):
		# 	is_toroidal.append(1)
		# else:
		# 	is_toroidal.append(0)

		# 이미지의 중심을 중심으로 이미지를 r_seta도 회전합니다.
		M = cv2.getRotationMatrix2D((cX, cY), r_seta, 1.0)	# cv2.getRotationMatrix2D(회전중심좌표(x,y 튜플), 회전각도, 스케일)
		rotated_seta = cv2.warpAffine(image, M, (h, w))	# cv2.warpAffine(src 원본이미지, M 아핀 맵 행렬, dsize 출력 이미지 크기) : 회전 변환을 계산

		# ####
		# #추가 실험
		# plt.imshow(rotated_seta)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
		# plt.show()
		#
		# ex = np.flip(rotated_seta, axis=0)
		# plt.imshow(ex)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
		# plt.show()
		#
		# ex = np.flip(rotated_seta,axis=1)
		# plt.imshow(ex)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
		# plt.show()
		# ####

		# 준비된 합성된 예제 이미지도 같이 회전시킴
		#rotated_exImg = cv2.warpAffine(exImg, M, (w, h))

		# ex = generateTextureMap(rotated_seta, blocksize, overlap, h, w, tolerance)
		# plt.imshow(ex)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
		# plt.show()

		######추가분
		mask_black = np.ones((h, w, 3))
		if(r_seta%90 != 0):
			def get_crosspt(y1, x21, y21, x22, y22):
				m2 = round((y22 - y21) / (x22 - x21) ,3)
				a = y1
				x1 = x21
				y1 = y21
				Y = a
				X = round( ((a-y1)/m2)+x1 ,3)

				return X, Y

			print("r_seta: {}".format(r_seta))
			a = w
			d = round(a*math.sqrt(2)/2, 3)
			de = math.radians(r_seta%90)
			r45 = math.radians(45)

			L1 = a//2
			L2s1x = d * round(math.cos(r45+de),3)
			L2s1y = d * round(math.sin(r45 + de),3)
			L2s2x = d * round(math.cos(r45-de),3)
			L2s2y = -1 * (d * round(math.sin(r45 - de),3))
			X2,Y2 = get_crosspt(L1, L2s1x, L2s1y, L2s2x, L2s2y)

			L3s1x = d * round(math.cos(r45 + de), 3)
			L3s1y = d * round(math.sin(r45 + de), 3)
			L3s2x = -1 * (d * round(math.cos(r45 - de), 3))
			L3s2y = d * round(math.sin(r45 - de), 3)
			X3, Y3 = get_crosspt(L1, L3s1x, L3s1y, L3s2x, L3s2y)

			Line1 = int(X3+L1)
			Line2 = int(X2-X3)
			Line3 = int(L1-X2)
			overplus = 3

			print(Line1,Line2,Line3)

			# fill_black_img = rotated_seta.copy()

			# 왼쪽 위 부분
			for x in range((Line1+1)+overplus):  # 원래대로라면 line_3 이 들어가야 하지만 검은 삼각형이 w,h가 같지않으므로 위에서부터 1칸씩 빼면서 내려가면 깔끔하게 삼각형이 안채워져서 원본이 정사각형이라는 가정 하(이것도 상관없는것같긴한데..)에 깔끔한 검은삼각형을 채우기 위하여 w,h가 같다고 가정하고 채워주기 위해 w=h 로 하였다..
				equation1 = ceil((-1) * (Line3 / Line1) * x + Line3)
				for y in range(equation1+overplus):
					# fill_black_img[y, x] = [1, 0, 0]
					mask_black[y, x] = 0
			# 왼쪽 아래 부분
			for x in range((Line3+1)+overplus):
				equation2 = ceil((Line1 / Line3) * x + (Line2 + Line3))
				for y in range(equation2-overplus,a):
					# fill_black_img[y, x] = [1, 0, 0]
					mask_black[y, x] = 0
			# 오른쪽 위 부분
			for x in range((a-Line3-1)-overplus,a):
				equation3 = ceil((Line1 / Line3) * (x - (Line1 + Line2)) + (Line2 + Line3) + (-1) * (Line2 + Line3))
				for y in range(equation3+overplus):
					# fill_black_img[y, x] = [1, 0, 0]
					mask_black[y, x] = 0
			# 오른쪽 아래 부분
			for x in range((a-Line1-1)-overplus,a):  # 원래대로라면 line_3 이 들어가야 하지만 검은 삼각형이 w,h가 같지않으므로 위에서부터 1칸씩 빼면서 내려가면 깔끔하게 삼각형이 안채워져서 원본이 정사각형이라는 가정 하(이것도 상관없는것같긴한데..)에 깔끔한 검은삼각형을 채우기 위하여 w,h가 같다고 가정하고 채워주기 위해 w=h 로 하였다..
				equation4 = ceil((-1) * (Line3 / Line1) * (x - (Line2 + Line3)) + Line3 + (a-Line3))
				for y in range(equation4-overplus,a):
					# fill_black_img[y, x] = [1, 0, 0]
					mask_black[y, x] = 0
			# pre_img = cv2.addWeighted(rotated_seta, 0.5, fill_black_img, 0.5, 0)

			# plt.imshow(pre_img)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
			# plt.show()  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌

		# ################


		# rotation -> 검은 삼각형 부분 => 합성 #########
		r_texture_black = r_generateTextureMap(rotated_seta, blocksize, overlap, h, w, tolerance, mask_black)	# 방향성 고려해서 새로 합성한 후보이미지

		# 어차피 회전 예제 이미지의 방향값을 가져오는 것이 목적이므로 더 자연스러운 새로만든 텍스쳐를 사용한다.
		# r_texture_black1 = r_texture_black[:h, :w, :]	# r_generateTextureMap () 함수 시 블록 사이즈에 나눠떨어지게 크기가 생성되므로 h,w 라도 좀 더 크게 잡힌다. 따라서 크기가 달라 아래에서 연산이 안되므로 조절해준다.
		# r_texture = rotated_seta * mask_black + r_texture_black1 * (1-mask_black)	# 기존 이미지 + 방향성 합성 이미지 검은부분용

		# plt.imshow(r_texture_black)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
		# plt.show()

		# img8.append(r_texture_black)
		# img8.append([rotated_seta,mask_black])
		img8.append([r_texture_black, mask_black])

		# Save
		pre_img = (255 * r_texture_black).astype(np.uint8)  # 최종 결과 텍스쳐 맵 -> 0~1, RGB 형태 => 원래대로로 돌림 (0~155 , BGR형태 , unit8)
		pre_img = cv2.cvtColor(pre_img, cv2.COLOR_RGB2BGR)

		cv2.imwrite("forcnn8" + str(i) + ".png", pre_img)
		#
		# pre_img1 = (255 * r_texture).astype(np.uint8)  # 최종 결과 텍스쳐 맵 -> 0~1, RGB 형태 => 원래대로로 돌림 (0~155 , BGR형태 , unit8)
		# pre_img1 = cv2.cvtColor(pre_img1, cv2.COLOR_RGB2BGR)
		#
		# cv2.imwrite("10img_" + str(i) + ".png", pre_img1)

	# #####################
	# #### gif 만드는 부분
	# gif_config = {
	# 	'loop': 1,  ## 0으로 세팅하면 무한 반복, 3으로 설정하면 3번 반복
	# 	'duration': 0.5  ## 다음 화면으로 넘어가는 시간
	# }

	# ## gif로 만들 이미지를 리스트로 만들어 줌
	# images = img8

	# ## mimwrite 대신 mimsave로도 가능
	# imageio.mimwrite(os.path.join(os.getcwd(), 'result_com.gif'),  ## 저장 경로
	# 					images,  ## 이미지 리스트
	# 					format='gif',  ## 저장 포맷
	# 					**gif_config  ## 부가 요소
	# 					)


	return img8

# 전처리 : 회전 이미지 생성
def simual_RotateExImg(image, blocksize, overlap, tolerance):  # 방향성 더해주기 위한 내가만든 함수
	print(">>Pre_roatateExImg")
	# 사용: generateTextureMap(image, block_size, overlap, outH, outW, args.tolerance)

	####try################################################
	# 이미지의 크기를 잡고 이미지의 중심을 계산합니다.
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)
	# is_toroidal = []

	img8 = []

	for i in range(1,ANGEL_NUM+1):
		imax = ANGEL_NUM
		r_seta = i / imax * 360

		# 이미지의 중심을 중심으로 이미지를 r_seta도 회전합니다.
		M = cv2.getRotationMatrix2D((cX, cY), r_seta, 1.0)	# cv2.getRotationMatrix2D(회전중심좌표(x,y 튜플), 회전각도, 스케일)
		rotated_seta = cv2.warpAffine(image, M, (h, w))	# cv2.warpAffine(src 원본이미지, M 아핀 맵 행렬, dsize 출력 이미지 크기) : 회전 변환을 계산

		######추가분
		mask_black = np.ones((h, w, 3))
		if(r_seta%90 != 0):
			def get_crosspt(y1, x21, y21, x22, y22):
				m2 = round((y22 - y21) / (x22 - x21) ,3)
				a = y1
				x1 = x21
				y1 = y21
				Y = a
				X = round( ((a-y1)/m2)+x1 ,3)

				return X, Y

			print("r_seta: {}".format(r_seta))
			a = w
			d = round(a*math.sqrt(2)/2, 3)
			de = math.radians(r_seta%90)
			r45 = math.radians(45)

			L1 = a//2
			L2s1x = d * round(math.cos(r45+de),3)
			L2s1y = d * round(math.sin(r45 + de),3)
			L2s2x = d * round(math.cos(r45-de),3)
			L2s2y = -1 * (d * round(math.sin(r45 - de),3))
			X2,Y2 = get_crosspt(L1, L2s1x, L2s1y, L2s2x, L2s2y)

			L3s1x = d * round(math.cos(r45 + de), 3)
			L3s1y = d * round(math.sin(r45 + de), 3)
			L3s2x = -1 * (d * round(math.cos(r45 - de), 3))
			L3s2y = d * round(math.sin(r45 - de), 3)
			X3, Y3 = get_crosspt(L1, L3s1x, L3s1y, L3s2x, L3s2y)

			Line1 = int(X3+L1)
			Line2 = int(X2-X3)
			Line3 = int(L1-X2)
			overplus = 3

			print(Line1,Line2,Line3)

			# fill_black_img = rotated_seta.copy()

			# 왼쪽 위 부분
			for x in range((Line1+1)+overplus):  # 원래대로라면 line_3 이 들어가야 하지만 검은 삼각형이 w,h가 같지않으므로 위에서부터 1칸씩 빼면서 내려가면 깔끔하게 삼각형이 안채워져서 원본이 정사각형이라는 가정 하(이것도 상관없는것같긴한데..)에 깔끔한 검은삼각형을 채우기 위하여 w,h가 같다고 가정하고 채워주기 위해 w=h 로 하였다..
				equation1 = ceil((-1) * (Line3 / Line1) * x + Line3)
				for y in range(equation1+overplus):
					# fill_black_img[y, x] = [1, 0, 0]
					mask_black[y, x] = 0
			# 왼쪽 아래 부분
			for x in range((Line3+1)+overplus):
				equation2 = ceil((Line1 / Line3) * x + (Line2 + Line3))
				for y in range(equation2-overplus,a):
					# fill_black_img[y, x] = [1, 0, 0]
					mask_black[y, x] = 0
			# 오른쪽 위 부분
			for x in range((a-Line3-1)-overplus,a):
				equation3 = ceil((Line1 / Line3) * (x - (Line1 + Line2)) + (Line2 + Line3) + (-1) * (Line2 + Line3))
				for y in range(equation3+overplus):
					# fill_black_img[y, x] = [1, 0, 0]
					mask_black[y, x] = 0
			# 오른쪽 아래 부분
			for x in range((a-Line1-1)-overplus,a):  # 원래대로라면 line_3 이 들어가야 하지만 검은 삼각형이 w,h가 같지않으므로 위에서부터 1칸씩 빼면서 내려가면 깔끔하게 삼각형이 안채워져서 원본이 정사각형이라는 가정 하(이것도 상관없는것같긴한데..)에 깔끔한 검은삼각형을 채우기 위하여 w,h가 같다고 가정하고 채워주기 위해 w=h 로 하였다..
				equation4 = ceil((-1) * (Line3 / Line1) * (x - (Line2 + Line3)) + Line3 + (a-Line3))
				for y in range(equation4-overplus,a):
					# fill_black_img[y, x] = [1, 0, 0]
					mask_black[y, x] = 0
			# pre_img = cv2.addWeighted(rotated_seta, 0.5, fill_black_img, 0.5, 0)

			# plt.imshow(pre_img)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
			# plt.show()  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌

		# ################
		
		img8.append([rotated_seta, mask_black])

		# Save
		pre_img = (255 * rotated_seta).astype(np.uint8)  # 최종 결과 텍스쳐 맵 -> 0~1, RGB 형태 => 원래대로로 돌림 (0~155 , BGR형태 , unit8)
		pre_img = cv2.cvtColor(pre_img, cv2.COLOR_RGB2BGR)

		cv2.imwrite("simual8" + str(i) + ".png", pre_img)

	return img8

###########################
# 거품 보간 시뮬레이션
def simual_findPatchHorizontal(refBlock, img8, img8_mask, blocksize, overlap, tolerance, tan_mask, blkIdx, where_white, where_black, where_mid):	# tolerance : 허용오차
	'''
	Find best horizontal match from the texture
	사용: findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance)
	'''
	print("horizontal")
	H, W = img8[0].shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
	errMat =  []

	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# openCV 경우 -> (rows, columns, channels) 튜플 보유,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		for r in range(len(img8)):
			# if (img8_mask[r][i:i + blocksize, j:j + blocksize] == 1).all():	# 회전 이미지의 이미지 유효값에서의 블록인 경우 만
			rmsVal = ((img8[r][i:i + blocksize, j:j + overlap] - refBlock[:,-overlap:]) ** 2).mean()  # (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균
			if rmsVal > 0:
				errMat.append([i, j, r, rmsVal])  # 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	y,x,r = 0,0,0
	# 마스크 1 안에 들어갈 경우
	if (tan_mask[:blocksize, (blkIdx):(blkIdx + blocksize)] == 1).any():  
		errWhite = []
		for ii in range(len(errMat)):
			for jj in range(len(where_white)):
				if(errMat[ii][:-1] == where_white[jj]):
					errWhite.append(errMat[ii])

		errWhite.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

		errIndex = []
		errIndex.append(errWhite[:5])  # 앞에 5개
		errIndex = sum(errIndex, [])  # [] 한꺼풀 벗겨주기

		# where_white2=[]
		# for k in range(len(where_white)):	# foam data 일정이상 하얀부분인 인덱스 중
		# 	if errMat[where_white[k][0],where_white[k][1]] < (1.0 + tolerance)*(minVal):	# err가 허용오차 값 이하일때의 인덱스 따로 빼둠
		# 		where_white2.append(where_white[k])
		# if len(where_white2)==0:	# 하얀부분의 해당위치의 err가 허용오차 값 이하인 블록이 존재하지 않을 때
		# 	for k in range(len(where_white)):  # foam data 일정이상 하얀부분인 인덱스 중
		# 		if errMat[where_white[k][0], where_white[k][1]] < (1.0 + 0.5) * (minVal):  # 허용 오차 늘려서 에러 다시 구함
		# 			where_white2.append(where_white[k])
		# 	if len(where_white2)==0:
		# 		for k in range(len(where_white)):  # foam data 일정이상 하얀부분인 인덱스 중
		# 			where_white2.append(where_white[k])

		while (True):
			c = np.random.randint(len(errIndex))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
			y = errIndex[c][0]
			x = errIndex[c][1]
			r = errIndex[c][2]
			if (img8_mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
				break

	# 마스크 2 안에 들어갈 경우
	elif (tan_mask[:blocksize, (blkIdx):(blkIdx + blocksize)] == 2).any():  
		errMid = []
		for ii in range(len(errMat)):
			for jj in range(len(where_mid)):
				if (errMat[ii][:-1] == where_mid[jj]):
					errMid.append(errMat[ii])

		errMid.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

		errIndex = []
		errIndex.append(errMid[:5])  # 앞에 5개
		errIndex = sum(errIndex, [])  # [] 한꺼풀 벗겨주기

		while (True):
			c = np.random.randint(len(errIndex))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
			y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
			if (img8_mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
				break

	# 마스크 0 밖 부분
	else:	
		errBlack = []
		for ii in range(len(errMat)):
			for jj in range(len(where_black)):
				if (errMat[ii][:-1] == where_black[jj]):
					errBlack.append(errMat[ii])

		errBlack.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

		errIndex = []
		errIndex.append(errBlack[:5])  # 앞에 5개
		errIndex = sum(errIndex, [])  # [] 한꺼풀 벗겨주기
		print("len: : {}".format(len(errIndex)))

		# where_black2 = []
		# for k in range(len(where_black)):  # foam data 일정이상 검은부분인 인덱스 중
		# 	if errMat[where_black[k][0], where_black[k][1]] < (1.0 + tolerance) * (
		# 			minVal):  # err가 허용오차 값 이하일때의 인덱스 따로 빼둠
		# 		where_black2.append(where_black[k])
		# if len(where_black2) == 0:  # 검은부분의 해당위치의 err가 허용오차 값 이하인 블록이 존재하지 않을 때
		# 	for k in range(len(where_black)):  # foam data 일정이상 하얀부분인 인덱스 중
		# 		if errMat[where_black[k][0], where_black[k][1]] < (1.0 + 0.5) * (minVal):  # 허용 오차 늘려서 에러 다시 구함
		#
		# 			where_black2.append(where_black[k])
		# 	if len(where_black2)==0:
		# 		for k in range(len(where_black)):  # foam data 일정이상 하얀부분인 인덱스 중
		# 			where_black2.append(where_black[k])

		while(True):
			c = np.random.randint(len(errIndex))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
			y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
			if (img8_mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
				break
		# return img8[r][y:y + blocksize, x:x + blocksize]  # 텍스쳐에서 해당 블록 return

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

	# plt.imshow(img8[r][y:y + blocksize, x:x + blocksize])  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
	# plt.show()

	return img8[r][y:y + blocksize, x:x + blocksize]	# 텍스쳐에서 해당 블록 return


# tan 추가 합성
def simual_findPatchBoth(refBlockLeft, refBlockTop, img8, img8_mask, blocksize, overlap, tolerance, tan_mask, blkIndexI, blkIndexJ, where_white, where_black, where_mid):
	'''
	Find best horizontal and vertical match from the texture
	사용: findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance)
	'''
	print("Both")
	H, W = img8[0].shape[:2]
	errMat = []

	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# [0,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		for r in range(len(img8)):
			if (img8_mask[r][i:i + blocksize, j:j + blocksize] == 1).all():  # 회전 이미지의 이미지 유효값에서의 블록인 경우 만
				rmsVal = ((img8[r][i:i+overlap, j:j+blocksize] - refBlockTop[-overlap:, :])**2).mean()	# (위의 이웃 블록의 오버랩 부분 - 각 블록의 위쪽 오버랩 부분) 제곱 의 평균
				rmsVal = rmsVal + ((img8[r][i:i+blocksize, j:j+overlap] - refBlockLeft[:, -overlap:])**2).mean()	# (왼쪽의 이웃 블록의 오버랩 부분 - 각 블록의 오른쪽 오버랩 부분) 제곱 의 평균

				if rmsVal > 0:
					errMat.append([i, j, r, rmsVal])	# 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	if (tan_mask[(blkIndexI):(blkIndexI + blocksize), (blkIndexJ):(blkIndexJ + blocksize)] == 1).any():  # 마스크 1 안에 들어갈 경우
		errWhite = []
		for ii in range(len(errMat)):
			for jj in range(len(where_white)):
				if (errMat[ii][:-1] == where_white[jj]):
					errWhite.append(errMat[ii])

		errWhite.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

		errIndex = []
		errIndex.append(errWhite[:5])  # 앞에 5개
		errIndex = sum(errIndex, [])  # [] 한꺼풀 벗겨주기

		while (True):
			c = np.random.randint(
				len(errIndex))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
			y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
			if (img8_mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
				break
	elif (tan_mask[(blkIndexI):(blkIndexI + blocksize), (blkIndexJ):(blkIndexJ + blocksize)] == 2).any():  # 마스크 2 안에 들어갈 경우
		errMid = []
		for ii in range(len(errMat)):
			for jj in range(len(where_mid)):
				if (errMat[ii][:-1] == where_mid[jj]):
					errMid.append(errMat[ii])

		errMid.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

		errIndex = []
		errIndex.append(errMid[:5])  # 앞에 5개
		errIndex = sum(errIndex, [])  # [] 한꺼풀 벗겨주기

		while (True):
			c = np.random.randint(len(errIndex))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
			y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
			if (img8_mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
				break
	else:  # 마스크 0 밖 부분
		errBlack = []
		for ii in range(len(errMat)):
			for jj in range(len(where_black)):
				if (errMat[ii][:-1] == where_black[jj]):
					errBlack.append(errMat[ii])

		errBlack.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

		errIndex = []
		errIndex.append(errBlack[:5])  # 앞에 5개
		errIndex = sum(errIndex, [])  # [] 한꺼풀 벗겨주기

		while (True):
			c = np.random.randint(len(errIndex))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
			y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
			if (img8_mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
				break

	return img8[r][y:y + blocksize, x:x + blocksize]  # 텍스쳐에서 해당 블록 return


# tan 추가 합성
def simual_findPatchVertical(refBlock, img8, img8_mask, blocksize, overlap, tolerance, tan_mask, blkIdx, where_white, where_black, where_mid):
	'''
	Find best vertical match from the texture
	사용: findPatchVertical(refBlock, image, blocksize, overlap, tolerance)
	'''
	print("Vertical")
	H, W = img8[0].shape[:2]  # 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
	errMat = []

	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# [0,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		for r in range(len(img8)):
			if (img8_mask[r][i:i + blocksize, j:j + blocksize] == 1).all():  # 회전 이미지의 이미지 유효값에서의 블록인 경우 만
				rmsVal = ((img8[r][i:i+overlap, j:j+blocksize] - refBlock[-overlap:, :]) ** 2).mean()  # (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균
				if rmsVal > 0:
					errMat.append([i, j, r, rmsVal])  # 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	y, x, r = 0, 0, 0
	if (tan_mask[(blkIdx):(blkIdx + blocksize), :blocksize] == 1).any():  # 마스크 1 안에 들어갈 경우
		errWhite = []
		for ii in range(len(errMat)):
			for jj in range(len(where_white)):
				# if (errMat[ii][0] == where_white[jj][0]) and (errMat[ii][1] == where_white[jj][1]):
				if (errMat[ii][:-1] == where_white[jj]):
					errWhite.append(errMat[ii])

		errWhite.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

		errIndex = []
		errIndex.append(errWhite[:5])  # 앞에 5개
		errIndex = sum(errIndex, [])  # [] 한꺼풀 벗겨주기

		while (True):
			c = np.random.randint(len(errIndex))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
			y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
			if (img8_mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
				break
	elif (tan_mask[(blkIdx):(blkIdx + blocksize), :blocksize] == 2).any():  # 마스크 2 안에 들어갈 경우
		errMid = []
		for ii in range(len(errMat)):
			for jj in range(len(where_mid)):
				# if (errMat[ii][0] == where_mid[jj][0]) and (errMat[ii][1] == where_mid[jj][1]) :
				if (errMat[ii][:-1] == where_mid[jj]):
					errMid.append(errMat[ii])

		errMid.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

		errIndex = []
		errIndex.append(errMid[:5])  # 앞에 5개
		errIndex = sum(errIndex, [])  # [] 한꺼풀 벗겨주기

		while (True):
			c = np.random.randint(len(errIndex))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
			y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
			if (img8_mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
				break
	else:  # 마스크 0 밖 부분

		errBlack = []
		for ii in range(len(errMat)):
			for jj in range(len(where_black)):
				# if (errMat[ii][0] == where_black[jj][0]) and (errMat[ii][1] == where_black[jj][1]):
				if (errMat[ii][:-1] == where_black[jj]):
					errBlack.append(errMat[ii])

		errBlack.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

		errIndex = []
		errIndex.append(errBlack[:5])  # 앞에 5개
		errIndex = sum(errIndex, [])  # [] 한꺼풀 벗겨주기

		while (True):
			c = np.random.randint(
				len(errIndex))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
			y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
			if (img8_mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
				break

	return img8[r][y:y + blocksize, x:x + blocksize]  # 텍스쳐에서 해당 블록 return

############################


#추가####################
# foam data 에 대한 합성
def foam_generateTextureMap(image, blocksize, overlap, outH, outW, tolerance, angle):	# main.py에서 사용되는 메인. tolerance : 허용요차
	# 사용: generateTextureMap(image, block_size, overlap, outH, outW, args.tolerance)
	# ceil() : 소수점 자리의 숫자를 무조건 올리는 함수
	nH = int(ceil((outH - blocksize)*1.0/(blocksize - overlap)))	# 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?
	nW = int(ceil((outW - blocksize)*1.0/(blocksize - overlap)))	# 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?

	textureMap = np.zeros(((blocksize + nH*(blocksize - overlap)), (blocksize + nW*(blocksize - overlap)), image.shape[2]))
	# [(H기준 : nH(들어가는 블록개수) * (오버랩 뺀 블록실제사이즈) + 마지막에 오버랩 안되므로 블록 하나 더 사이즈) , (W기준 동일) , 색상] => 0으로 초기화
	# Starting index and block
	H, W = image.shape[:2]

	pre_img8 = simual_RotateExImg(image, blocksize, overlap, tolerance)  # pre_img8 : [ [rotated_seta , mask] , [rotated_seta , mask] , .. ]
	# => shape : (8, 2, h, w, 3)
	tmp_img8 = list(zip(*pre_img8))  # [ [rotated_seta 끼리 ] , [mask 끼리] ] 로 형태 변환
	img8 = tmp_img8[0]
	img8_mask = tmp_img8[1]

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
	c, d = a//2, b//2
	tan_mask = np.zeros((a,b))

	slope = 0	# 회전된 직선영역의 기울기
	is_90 = False	# flag : 90도인가, 90도일경우에만 직선의 방정식 x= a 꼴이기 때문

	if angle%90==0:
		if angle%180==0:	# 180도 일 경우 y = y1 꼴
			slope = 0
		else:	# 90도 일 경우 x = x1 꼴
			is_90 = True
	else:	# 90, 180 도 배수 제외한 나머지 일 경우 y = ax + b
		slope = math.tan(math.radians(angle))

	flagi = 0
	flagj = 0

	tmpj = 0
	for y in range(a):	# h
		t = 0
		for x in range(b):	# w
			if is_90 == True:	# 각도 90도일 경우 특수 : x = d 꼴 / 나머지 : y = ~ 꼴
				tan_line = d

				if tan_line-30<=x and x<=tan_line+30:	# 기울어진 직선에서 얼만큼 두께를 줄 것인지
					tan_mask[y,x] = 1
					t+=1
					textureMap[y,x]=(255,0,0)
				elif tan_line-50<=x and x<=tan_line+50:
					tan_mask[y,x] = 2
					t+=1
					textureMap[y,x]=(0,255,0)

			else:
				tan_line = (a-1) - (math.ceil(slope * (x - d)) + c )	# 정해진 각도를 기울기로 갖는 이미지 상 직선

				if tan_line-30<=y and y<=tan_line+30:	# 기울어진 직선에서 얼만큼 두께를 줄 것인지
					tan_mask[y,x] = 1
					t+=1
					textureMap[y,x]=(255,0,0)
				elif tan_line-40<=y and y<=tan_line+40:	# 자연스러운 분포를 위해 겉에 한겹 더
					tan_mask[y,x] = 2
					t+=1
					textureMap[y,x]=(0,255,0)
	print("line generate finished")

		# if t==0:	# for 문  y -> x 순으로 확인할 때 tan_line 이 짝수가 나오는 식이면 y 가 홀수일때 조건 만족하는 x를 찾을 수 없기 때문에 이전값을 저장했다가 그대로 씀
		# 	tan_mask[y,tmpj] = 1
		# 	textureMap[y, tmpj] = (255, 0, 0)

	# plt.imshow(textureMap)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
	# plt.show()
	
	# textureMap[flagi:flagi+blocksize, flagj:flagj+blocksize, :] = startBlock  # 0으로 초기화된 맵에서 첫번째 블록에 랜덤하게 가져온 블록 대입함
	######################

	where_white = []
	where_mid = []
	where_black = []

	for r in range(len(img8)):
		print("블록마다 유효값 계산 중")
		for i in range(0, H-blocksize):
			for j in range(0,W-blocksize):
				if (img8_mask[r][i:i + blocksize, j:j + blocksize] == 1).all():
					count_black = 0
					# 한 블록 당 검은 부분 얼만큼?
					for si in range(i, i + blocksize):
						for sj in range(j, j + blocksize):
							if (img8[r][si, sj] == [0, 0, 0]).all():
								count_black += 1
					# 검은 부분의 정도에 따라 나누기
					if count_black <= (blocksize * blocksize * (1 / 3)):
						where_white.append([i, j, r])
					elif count_black <= (blocksize * blocksize * (2 / 3)):
						where_mid.append([i, j, r])
					elif count_black >= (blocksize * blocksize * (2 / 3)):
						where_black.append([i, j, r])
	print("블록 유효값 계산 완료")

	################################

	# 이제 만들어갈 texturemap 의 첫 블록 - 랜덤하게 끼워넣음

	print("rand first block")
	randH = np.random.randint(H - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값
	randW = np.random.randint(W - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값


	startBlock = image[randH:randH + blocksize, randW:randW + blocksize]  # 랜덤한 위치에서 시작하는 블록 사이즈만큼 잘라서 가져옴
	textureMap[:blocksize, :blocksize, :] = startBlock  # 0으로 초기화된 맵에서 첫번째 블록에 랜덤하게 가져온 블록 대입함

	print("generate start>>")
	# Fill the first row : 행(아래 위)
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[1]-overlap, (blocksize-overlap))):	# enumerate() : 인덱스와 원소 차례로 반환
		# 오버랩 부분 제외 블록 부분부터 ~ 오버랩 제외 열들까지 , 오버랩 제외한 블록사이즈만큼 옆으로 이동 (오른 -> 왼)

		# Find horizontal error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		# blkIdx = 블록에서 오버랩 되는 부분 시작점 인덱스
		refBlock = textureMap[:blocksize, (blkIdx-blocksize+overlap):(blkIdx+overlap)]	#texturemap 의 한줄제외 모든 행에 대하여 열단위로 블록 한 칸만큼 계속 이동하면서 대입
		patchBlock = simual_findPatchHorizontal(refBlock, img8, img8_mask, blocksize, overlap, tolerance, tan_mask, blkIdx, where_white, where_black, where_mid)	# 미리 만든 패치 찾는 함수
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
		patchBlock = simual_findPatchVertical(refBlock, img8, img8_mask, blocksize, overlap, tolerance, tan_mask, blkIdx, where_white, where_black, where_mid)	# 미리 만든 패치 찾는 함수
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

			patchBlock = simual_findPatchBoth(refBlockLeft, refBlockTop, img8, img8_mask, blocksize, overlap, tolerance, tan_mask, blkIndexI, blkIndexJ, where_white, where_black, where_mid)
			minCutPatch = getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap)

			textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ):(blkIndexJ+blocksize)] = minCutPatch

			# refBlockLeft = 0.5
			# textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ-blocksize+overlap):(blkIndexJ+overlap)] = refBlockLeft
			# textureMap[(blkIndexI-blocksize+overlap):(blkIndexI+overlap), (blkIndexJ):(blkIndexJ+blocksize)] = [0.5, 0.6, 0.7]
			# break
		print("{} out of {} rows complete...".format(i+1, nH+1))
		# break

	return textureMap
#############################
#############################
#############################


#추가#########################
# 회전 입력 텍스처 추가 합성부분
def r_findPatchHorizontal(refBlock, texture, blocksize, overlap, tolerance, mask):	# tolerance : 허용오차
	'''
	Find best horizontal match from the texture
	사용: findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance)
	'''
	H, W = texture[0].shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
	errMat = []
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# openCV 경우 -> (rows, columns, channels) 튜플 보유,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		for r in range(len(texture)):
			if (mask[r][i:i + blocksize, j:j + blocksize] == 1).all():
				rmsVal = ((texture[r][i:i + blocksize, j:j + overlap] - refBlock[:, -overlap:]) ** 2).mean()  # (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균
				if rmsVal > 0:
					errMat.append([i,j,r,rmsVal]) # 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	errMat.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

	errIndex = []
	errIndex.append(errMat[:5])
	errIndex = sum(errIndex, [])

	while (True):
		c = np.random.randint(len(errIndex))	# random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
		y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
		if (mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
			break

	return texture[r][y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return


# 회전 입력 텍스처 추가 합성부분
def r_findPatchBoth(refBlockLeft, refBlockTop, texture, blocksize, overlap, tolerance, mask):
	'''
	Find best horizontal and vertical match from the texture
	사용: findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance)
	'''
	H, W = texture[0].shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
								# openCV 경우 -> (rows, columns, channels) 튜플 보유
	errMat = []
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# [0,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		for r in range(len(texture)):
			if (mask[r][i:i+blocksize, j:j+blocksize] == 1).all():
				rmsVal = ((texture[r][i:i+overlap, j:j+blocksize] - refBlockTop[-overlap:, :])**2).mean()	# (위의 이웃 블록의 오버랩 부분 - 각 블록의 위쪽 오버랩 부분) 제곱 의 평균
				rmsVal = rmsVal + ((texture[r][i:i+blocksize, j:j+overlap] - refBlockLeft[:, -overlap:])**2).mean()	# (왼쪽의 이웃 블록의 오버랩 부분 - 각 블록의 오른쪽 오버랩 부분) 제곱 의 평균
				if rmsVal > 0:
					errMat.append([i,j,r,rmsVal])

	errMat.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

	errIndex = []
	errIndex.append(errMat[:5])
	errIndex = sum(errIndex, [])

	while (True):
		c = np.random.randint(len(errIndex))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
		y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
		if (mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
			break

	return texture[r][y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return


# 회전 입력 텍스처 추가 합성부분
def r_findPatchVertical(refBlock, texture, blocksize, overlap, tolerance, mask):
	'''
	Find best vertical match from the texture
	사용: findPatchVertical(refBlock, image, blocksize, overlap, tolerance)
	'''
	H, W = texture[0].shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
								# openCV 경우 -> (rows, columns, channels) 튜플 보유

	errMat = []
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# [0,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		for r in range(len(texture)):
			if (mask[r][i:i+blocksize, j:j+blocksize] == 1).all():
				rmsVal = ((texture[r][i:i+overlap, j:j+blocksize] - refBlock[-overlap:, :])**2).mean()	# (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균
				if rmsVal > 0:
					errMat.append([i,j,r,rmsVal])	# 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	errMat.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

	errIndex = []
	errIndex.append(errMat[:5])
	errIndex = sum(errIndex, [])

	while (True):
		c = np.random.randint(len(errIndex))	# random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
		y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
		if (mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
			break

	return texture[r][y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return
#############################

#추가###################
# 회전 입력 텍스처 추가 합성부분
def r_generateTextureMap(image, blocksize, overlap, y, x, tolerance, mask):	# 회전이미지에서 검은부분 합성으로 채우기
	print(">>r_generateTextureMap")
	# 사용: generateTextureMap(image, block_size, overlap, outH, outW, args.tolerance)
	# ceil() : 소수점 자리의 숫자를 무조건 올리는 함수
	nH = int(ceil((y - blocksize) * 1.0 / (blocksize - overlap)))  # 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?
	nW = int(ceil((x - blocksize) * 1.0 / (blocksize - overlap)))  # 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?
	textureMap = np.zeros(((blocksize + nH * (blocksize - overlap)), (blocksize + nW * (blocksize - overlap)), image.shape[2]))
	# [(H기준 : nH(들어가는 블록개수) * (오버랩 뺀 블록실제사이즈) + 마지막에 오버랩 안되므로 블록 하나 더 사이즈) , (W기준 동일) , 색상] => 0으로 초기화

	# patch 비교 : image + 상하반전 image 두개 사용
	two_img = []
	two_mask = []

	two_img.append(image)
	two_img.append(np.flip(image, axis=0))
	two_img.append(np.flip(image, axis=1))

	two_mask.append(mask)
	two_mask.append(np.flip(mask, axis=0))
	two_mask.append(np.flip(mask, axis=1))

	# Starting index and block
	H, W = image.shape[:2]

	while (True):  # do-while 문
		randH = np.random.randint(H - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값
		randW = np.random.randint(W - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값

		if (mask[randH:randH + blocksize, randW:randW + blocksize] == 1).all():  # 로테이션 이미지 존재하는 부분일 때의 random 값 뽑아내기
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
		patchBlock = r_findPatchHorizontal(refBlock, two_img, blocksize, overlap, tolerance, two_mask)  # 미리 만든 패치 찾는 함수
		minCutPatch = getMinCutPatchHorizontal(refBlock, patchBlock, blocksize, overlap)  # 미리 만든 최소 경로 찾는 함수
		textureMap[:blocksize, (blkIdx):(blkIdx + blocksize)] = minCutPatch  # 오버랩부분 경계선 최소경로로 자름
	print("pre rotate => {} out of {} rows complete...".format(1, nH + 1))

	### Fill the first column 열 (오른 왼쪽)
	for i, blkIdx in enumerate(range((blocksize - overlap), textureMap.shape[0] - overlap, (blocksize - overlap))):  # # enumerate() : 인덱스와 원소 차례로 반환
		# 오버랩 부분 제외 블록 부분부터 ~ 오버랩 제외 행들까지 , 오버랩 제외한 블록사이즈만큼 옆으로 이동 (위 -> 아래)

		# Find vertical error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		# blkIdx = 블록에서 오버랩 되는 부분 시작점 인덱스
		refBlock = textureMap[(blkIdx - blocksize + overlap):(blkIdx + overlap), :blocksize]  # texturemap 의 한줄제외 모든 열에 대하여 행단위로 블록 한 칸만큼 계속 이동하면서 대입
		patchBlock = r_findPatchVertical(refBlock, two_img, blocksize, overlap, tolerance, two_mask)  # 미리 만든 패치 찾는 함수
		minCutPatch = getMinCutPatchVertical(refBlock, patchBlock, blocksize, overlap)  # 미리 만든 최소 경로 찾는 함수
		textureMap[(blkIdx):(blkIdx + blocksize), :blocksize] = minCutPatch  # 오버랩부분 경계선 최소경로로 자름
	print("pre rotate => {} out of {} rows complete...".format(2, nH + 1))

	### Fill in the other rows and columns
	for i in range(1, nH + 1):
		for j in range(1, nW + 1):
			# Choose the starting index for the texture placement
			blkIndexI = i * (blocksize - overlap)
			blkIndexJ = j * (blocksize - overlap)
			# Find the left and top block, and the min errors independently
			refBlockLeft = textureMap[(blkIndexI):(blkIndexI + blocksize), (blkIndexJ - blocksize + overlap):(blkIndexJ + overlap)]
			refBlockTop = textureMap[(blkIndexI - blocksize + overlap):(blkIndexI + overlap), (blkIndexJ):(blkIndexJ + blocksize)]

			patchBlock = r_findPatchBoth(refBlockLeft, refBlockTop, two_img, blocksize, overlap, tolerance, two_mask)
			# if(i>nH-1):
			# 	plt.imshow(patchBlock)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
			# 	plt.show()
			minCutPatch = getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap)

			textureMap[(blkIndexI):(blkIndexI + blocksize), (blkIndexJ):(blkIndexJ + blocksize)] = minCutPatch

		# refBlockLeft = 0.5
		# textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ-blocksize+overlap):(blkIndexJ+overlap)] = refBlockLeft
		# textureMap[(blkIndexI-blocksize+overlap):(blkIndexI+overlap), (blkIndexJ):(blkIndexJ+blocksize)] = [0.5, 0.6, 0.7]
		# break
		print("pre rotate => {} out of {} rows complete...".format(i + 1, nH + 1))
	# break

	return textureMap
#######################

# 추가##################


def fin_findPatchHorizontal(refBlock, img8, blocksize, overlap, tolerance, mask):	# tolerance : 허용오차
	'''
	Find best horizontal match from the texture
	사용: findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance)
	'''
	H, W = img8[0].shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출

	errMat = []
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# openCV 경우 -> (rows, columns, channels) 튜플 보유,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		for r in range(len(img8)):
			if (mask[r][i:i + blocksize, j:j + blocksize] == 1).all():
				rmsVal = ((img8[r][i:i+blocksize, j:j+overlap] - refBlock[:, -overlap:])**2).mean()	# (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균
				if rmsVal > 0:
					errMat.append([i,j,r,rmsVal]) # 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	# errMat_2dLow = list(zip(*errMat))
	# minVal = np.min(errMat_2dLow[3])	# 에러범위 값 중 가장 작은 것

	errMat.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

	errIndex = []
	errIndex.append(errMat[:5])
	errIndex = sum(errIndex, [])

	# errIndex = []
	# for i in range(len(errMat)):
	# 	if errMat[i][3] < (1.0 + tolerance) * (minVal):
	# 		errIndex.append(errMat[i])

	while (True):
		c = np.random.randint(len(errIndex))	# random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
		y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
		if (mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
			break

	return img8[r][y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return



def fin_findPatchBoth(refBlockLeft, refBlockTop, img8, blocksize, overlap, tolerance, mask):
	'''
	Find best horizontal and vertical match from the texture
	사용: findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance)
	'''
	H, W = img8[0].shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
								# openCV 경우 -> (rows, columns, channels) 튜플 보유
	errMat = []

	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# [0,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		for r in range(len(img8)):
			if (mask[r][i:i + blocksize, j:j + blocksize] == 1).all():
				rmsVal = ((img8[r][i:i+overlap, j:j+blocksize] - refBlockTop[-overlap:, :])**2).mean()	# (위의 이웃 블록의 오버랩 부분 - 각 블록의 위쪽 오버랩 부분) 제곱 의 평균
				rmsVal = rmsVal + ((img8[r][i:i+blocksize, j:j+overlap] - refBlockLeft[:, -overlap:])**2).mean()	# (왼쪽의 이웃 블록의 오버랩 부분 - 각 블록의 오른쪽 오버랩 부분) 제곱 의 평균
				if rmsVal > 0:
					errMat.append([i, j, r, rmsVal])  # 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	# errMat_2dLow = list(zip(*errMat))
	# minVal = np.min(errMat_2dLow[3])  # 에러범위 값 중 가장 작은 것

	errMat.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

	errIndex = []
	errIndex.append(errMat[:5])
	errIndex = sum(errIndex, [])

	# errIndex = []
	# for i in range(len(errMat)):
	# 	if errMat[i][3] < (1.0 + tolerance) * (minVal):
	# 		errIndex.append(errMat[i])

	while (True):
		c = np.random.randint(len(errIndex))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
		y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
		if (mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
			break

	return img8[r][y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return



def fin_findPatchVertical(refBlock, img8, blocksize, overlap, tolerance, mask):
	'''
	Find best vertical match from the texture
	사용: findPatchVertical(refBlock, image, blocksize, overlap, tolerance)
	'''
	H, W = img8[0].shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
								# openCV 경우 -> (rows, columns, channels) 튜플 보유
	errMat =  []
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# [0,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		for r in range(len(img8)):
			# if (mask[r][i:i + blocksize, j:j + blocksize] == 1).all():
			rmsVal = ((img8[r][i:i+overlap, j:j+blocksize] - refBlock[-overlap:, :])**2).mean()	# (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균
			if rmsVal > 0:
				errMat.append([i,j,r,rmsVal])

	# errMat_2dLow = list(zip(*errMat))

	# minVal = np.min(errMat_2dLow[3])  # 에러범위 값 중 가장 작은 것

	errMat.sort(key=lambda x:x[3])	# err 작은것부터 오름차순 정렬

	errIndex = []
	errIndex.append(errMat[:5])
	errIndex = sum(errIndex, [])	# errIndex 에서 [] 한꺼풀 벗겨줌
	# for i in range(len(errMat)):
	# 	if errMat[i][3] < (1.0 + tolerance) * (minVal):
	# 		errIndex.append(errMat[i])

	# while (True):
	c = np.random.randint(len(errIndex))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
	y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
		# if (mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
		# 	break

	return img8[r][y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return


def fin_generateTextureMap(image, blocksize, overlap, outH, outW, tolerance):	# main.py에서 사용되는 메인. tolerance : 허용요차
	print(">>fin_generateTextureMap")
	# 사용: generateTextureMap(image, block_size, overlap, outH, outW, args.tolerance)
	# ceil() : 소수점 자리의 숫자를 무조건 올리는 함수
	nH = int(ceil((outH - blocksize)*1.0/(blocksize - overlap)))	# 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?
	nW = int(ceil((outW - blocksize)*1.0/(blocksize - overlap)))	# 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?

	textureMap = np.zeros(((blocksize + nH*(blocksize - overlap)), (blocksize + nW*(blocksize - overlap)), image.shape[2]))
	# [(H기준 : nH(들어가는 블록개수) * (오버랩 뺀 블록실제사이즈) + 마지막에 오버랩 안되므로 블록 하나 더 사이즈) , (W기준 동일) , 색상] => 0으로 초기화
	# Starting index and block
	H, W = image.shape[:2]
	pre_img8 = Make_RotateExImg(image, blocksize, overlap, tolerance)	# pre_img8 : [ [rotated_seta , mask] , [rotated_seta , mask] , .. ]
																		# => shape : (8, 2, h, w, 3)
	tmp_img8 = list(zip(*pre_img8))	# [ [rotated_seta 끼리 ] , [mask 끼리] ] 로 형태 변환
	img8 = tmp_img8[0]
	mask = tmp_img8[1]

	randH = np.random.randint(H - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값
	randW = np.random.randint(W - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값

	startBlock = image[randH:randH+blocksize, randW:randW+blocksize]	# 랜덤한 위치에서 시작하는 블록 사이즈만큼 잘라서 가져옴
	textureMap[:blocksize, :blocksize, :] = startBlock	# 0으로 초기화된 맵에서 첫번째 블록에 랜덤하게 가져온 블록 대입함

	print(">>fin_generateTextureMap 패치이어붙이기 시작")
	# Fill the first row : 행(아래 위)
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[1]-overlap, (blocksize-overlap))):	# enumerate() : 인덱스와 원소 차례로 반환
		# 오버랩 부분 제외 블록 부분부터 ~ 오버랩 제외 열들까지 , 오버랩 제외한 블록사이즈만큼 옆으로 이동 (오른 -> 왼)

		# Find horizontal error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		# blkIdx = 블록에서 오버랩 되는 부분 시작점 인덱스
		refBlock = textureMap[:blocksize, (blkIdx-blocksize+overlap):(blkIdx+overlap)]	#texturemap 의 한줄제외 모든 행에 대하여 열단위로 블록 한 칸만큼 계속 이동하면서 대입
		patchBlock = fin_findPatchHorizontal(refBlock, img8, blocksize, overlap, tolerance, mask)	# 미리 만든 패치 찾는 함수
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
		patchBlock = fin_findPatchVertical(refBlock, img8, blocksize, overlap, tolerance, mask)	# 미리 만든 패치 찾는 함수
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

			patchBlock = fin_findPatchBoth(refBlockLeft, refBlockTop, img8, blocksize, overlap, tolerance, mask)
			minCutPatch = getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap)

			textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ):(blkIndexJ+blocksize)] = minCutPatch

			# refBlockLeft = 0.5
			# textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ-blocksize+overlap):(blkIndexJ+overlap)] = refBlockLeft
			# textureMap[(blkIndexI-blocksize+overlap):(blkIndexI+overlap), (blkIndexJ):(blkIndexJ+blocksize)] = [0.5, 0.6, 0.7]
			# break
		print("{} out of {} rows complete...".format(i+1, nH+1))

	return textureMap


# old code ###############
# 기존 회전예제 코드
def old_RotateExImg(image):  # 기존 이미지 8번 회전된 이미지로 만드는 기존 함수
	####try################################################
	# 이미지의 크기를 잡고 이미지의 중심을 계산합니다.
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)
	img8 = []

	for i in range(1, 9):
		imax = 8
		r_seta = i / imax * 360

		# 이미지의 중심을 중심으로 이미지를 r_seta도 회전합니다.
		M = cv2.getRotationMatrix2D((cX, cY), r_seta, 1.0)  # cv2.getRotationMatrix2D(회전중심좌표(x,y 튜플), 회전각도, 스케일)
		rotated_seta = cv2.warpAffine(image, M, (h, w))  # cv2.warpAffine(src 원본이미지, M 아핀 맵 행렬, dsize 출력 이미지 크기) : 회전 변환을 계산

		# 검은 부분 : sin cos  로 계산 경우
		L = w
		sin_seta = math.sin(r_seta % 90)
		cos_seta = math.cos(r_seta % 90)
		t = L / (sin_seta + cos_seta + 1)

		line_1 = ceil(t * sin_seta)  # 넉넉하게 라인 길이를 잡아줘야 하므로 올림으로 하였다.
		line_2 = ceil(t)
		line_3 = ceil(t * cos_seta)

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

		if r_seta%90 == 0:	#toroidal - tyling
			# print("toroidal 1")
			# plt.imshow(rotated_seta)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
			# plt.show()
			tNew = rotated_seta
			# tNew = np.flip(rotated_seta,axis=0)	# 상하반전
			# print("1shape:{}".format(tNew.shape))
			# plt.imshow(rotated_seta)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
			# plt.show()
		else:	#non toroidal - mirraring
			# print("non 1")
			# plt.imshow(rotated_seta)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
			# plt.show()
			ntNew = rotated_seta
			ntNew = np.flip(ntNew,axis=1)	# 좌우반전
			# print("2shape:{}".format(ntNew.shape))
			# plt.imshow(rotated_seta)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
			# plt.show()

		## 수정1
		mask_black = np.ones((h, w, 3))
		black_h = line_1  # 검은 삼각형 높이부분 : tsin@
		black_w = line_1  # 검은 삼각형 밑변부분 : tcos@

		# rotation -> 검은 삼각형 부분 => 마스크 만들기
		# 왼쪽 위 부분
		tmp = black_h
		for y in range(black_w):  # 원래대로라면 line_3 이 들어가야 하지만 검은 삼각형이 w,h가 같지않으므로 위에서부터 1칸씩 빼면서 내려가면 깔끔하게 삼각형이 안채워져서 원본이 정사각형이라는 가정 하(이것도 상관없는것같긴한데..)에 깔끔한 검은삼각형을 채우기 위하여 w,h가 같다고 가정하고 채워주기 위해 w=h 로 하였다..
			for x in range(tmp):
				mask_black[y, x] = 0
				if r_seta%90 == 0:	#toro
					rotated_seta[y,x]= tNew[y - (h - line_1), x - (w - line_1)]
				else:	#ntoro
					rotated_seta[y, x] = ntNew[y - (h - line_1), x - (w - line_1)]
			tmp -= 1

		# 왼쪽 아래 부분
		tmp = 1
		for y in range(L - black_h, L):
			for x in range(tmp):
				mask_black[y, x] = 0
				if r_seta%90 == 0:	#toro
					rotated_seta[y,x]= tNew[h - y, x - (w - line_1)]
				else:	#ntoro
					rotated_seta[y, x] = ntNew[h - y, x - (w - line_1)]
			tmp += 1
		# 오른쪽 위 부분
		tmp = black_w
		for y in range(black_h):
			for x in range(L - tmp, L):
				mask_black[y, x] = 0
				if r_seta%90 == 0:	#toro
					rotated_seta[y,x]= tNew[y - (h - line_1), w-x]
				else:	#ntoro
					rotated_seta[y, x] = ntNew[y - (h - line_1), w-x]
			tmp -= 1
		# 오른쪽 아래 부분
		tmp = 0
		for y in range(L - black_h,L):  # 원래대로라면 line_3 이 들어가야 하지만 검은 삼각형이 w,h가 같지않으므로 위에서부터 1칸씩 빼면서 내려가면 깔끔하게 삼각형이 안채워져서 원본이 정사각형이라는 가정 하(이것도 상관없는것같긴한데..)에 깔끔한 검은삼각형을 채우기 위하여 w,h가 같다고 가정하고 채워주기 위해 w=h 로 하였다..
			for x in range(L - tmp, L):
				mask_black[y, x] = 0
				if r_seta%90 == 0:	#toro
					rotated_seta[y,x]= tNew[h-y, w-x]
				else:	#ntoro
					rotated_seta[y, x] = ntNew[h-y, w-x]
			tmp += 1

		# rotation -> 검은 삼각형 부분 => 합성 #########
		# r_texture_black = r_generateTextureMap(rotated_seta, blocksize, overlap, h, w, tolerance, mask_black)	# 방향성 고려해서 새로 합성한 후보이미지

		# 어차피 회전 예제 이미지의 방향값을 가져오는 것이 목적이므로 더 자연스러운 새로만든 텍스쳐를 사용한다.
		# r_texture_black1 = r_texture_black[:h, :w, :]	# r_generateTextureMap () 함수 시 블록 사이즈에 나눠떨어지게 크기가 생성되므로 h,w 라도 좀 더 크게 잡힌다. 따라서 크기가 달라 아래에서 연산이 안되므로 조절해준다.
		# r_texture = rotated_seta * mask_black + r_texture_black1 * (1-mask_black)	# 기존 이미지 + 방향성 합성 이미지 검은부분용

		# plt.imshow(r_texture_black)  # array의 값들을 색으로 환산해 이미지의 형태로 보여줌
		# plt.show()

		# img8.append(r_texture_black)

		# Save
		pre_img = (255 * rotated_seta).astype(np.uint8)  # 최종 결과 텍스쳐 맵 -> 0~1, RGB 형태 => 원래대로로 돌림 (0~155 , BGR형태 , unit8)
		pre_img = cv2.cvtColor(pre_img, cv2.COLOR_RGB2BGR)

		cv2.imwrite("8img_" + str(i) + ".png", pre_img)
		img8.append([rotated_seta, mask_black])

	# # Save
	# pre_img = (255 * rotated_seta).astype(np.uint8)  # 최종 결과 텍스쳐 맵 -> 0~1, RGB 형태 => 원래대로로 돌림 (0~155 , BGR형태 , unit8)
	# pre_img = cv2.cvtColor(pre_img, cv2.COLOR_RGB2BGR)
	#
	# cv2.imwrite("8img_" + str(i) + ".png", pre_img)
	#
	# pre_img1 = (255 * r_texture).astype(np.uint8)  # 최종 결과 텍스쳐 맵 -> 0~1, RGB 형태 => 원래대로로 돌림 (0~155 , BGR형태 , unit8)
	# pre_img1 = cv2.cvtColor(pre_img1, cv2.COLOR_RGB2BGR)
	#
	# cv2.imwrite("10img_" + str(i) + ".png", pre_img1)

	return img8

def old_findPatchHorizontal(refBlock, img8, blocksize, overlap, tolerance, mask):	# tolerance : 허용오차
	'''
	Find best horizontal match from the texture
	사용: findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance)
	'''
	H, W = img8[0].shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출

	errMat = []
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# openCV 경우 -> (rows, columns, channels) 튜플 보유,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		for r in range(len(img8)):
			# if (mask[r][i:i + blocksize, j:j + blocksize] == 1).all():
			rmsVal = ((img8[r][i:i+blocksize, j:j+overlap] - refBlock[:, -overlap:])**2).mean()	# (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균
			if rmsVal > 0:
				errMat.append([i,j,r,rmsVal]) # 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	# errMat_2dLow = list(zip(*errMat))
	# minVal = np.min(errMat_2dLow[3])	# 에러범위 값 중 가장 작은 것

	errMat.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

	errIndex = []
	errIndex.append(errMat[:5])
	errIndex = sum(errIndex, [])

	# errIndex = []
	# for i in range(len(errMat)):
	# 	if errMat[i][3] < (1.0 + tolerance) * (minVal):
	# 		errIndex.append(errMat[i])

	# while (True):
	c = np.random.randint(len(errIndex))	# random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
	y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
		# if (mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
		# 	break

	return img8[r][y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return



def old_findPatchBoth(refBlockLeft, refBlockTop, img8, blocksize, overlap, tolerance, mask):
	'''
	Find best horizontal and vertical match from the texture
	사용: findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance)
	'''
	H, W = img8[0].shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
								# openCV 경우 -> (rows, columns, channels) 튜플 보유
	errMat = []

	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# [0,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		for r in range(len(img8)):
			# if (mask[r][i:i + blocksize, j:j + blocksize] == 1).all():
			rmsVal = ((img8[r][i:i+overlap, j:j+blocksize] - refBlockTop[-overlap:, :])**2).mean()	# (위의 이웃 블록의 오버랩 부분 - 각 블록의 위쪽 오버랩 부분) 제곱 의 평균
			rmsVal = rmsVal + ((img8[r][i:i+blocksize, j:j+overlap] - refBlockLeft[:, -overlap:])**2).mean()	# (왼쪽의 이웃 블록의 오버랩 부분 - 각 블록의 오른쪽 오버랩 부분) 제곱 의 평균
			if rmsVal > 0:
				errMat.append([i,j,r,rmsVal])	# 텍스쳐 크기에서 블록사이즈만큼 한줄 작아진 배열에 대입

	# errMat_2dLow = list(zip(*errMat))
	# minVal = np.min(errMat_2dLow[3])  # 에러범위 값 중 가장 작은 것

	errMat.sort(key=lambda x: x[3])  # err 작은것부터 오름차순 정렬

	errIndex = []
	errIndex.append(errMat[:5])
	errIndex = sum(errIndex, [])

	# errIndex = []
	# for i in range(len(errMat)):
	# 	if errMat[i][3] < (1.0 + tolerance) * (minVal):
	# 		errIndex.append(errMat[i])

	# while (True):
	c = np.random.randint(len(errIndex))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
	y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
		# if (mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
		# 	break

	return img8[r][y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return



def old_findPatchVertical(refBlock, img8, blocksize, overlap, tolerance, mask):
	'''
	Find best vertical match from the texture
	사용: findPatchVertical(refBlock, image, blocksize, overlap, tolerance)
	'''
	H, W = img8[0].shape[:2]	# 튜플 압축 풀기 -> 해당 texture 의 rows, columns  값 추출
								# openCV 경우 -> (rows, columns, channels) 튜플 보유
	errMat =  []
	for i, j in product(range(H-blocksize), range(W-blocksize)):	# product : 중복 순열 , 데이터를 뽑아 일렬로 나열하는 모든 경우의 수 / range : 0~해당 값까지
																	# [0,1,2, ... ,H-blocksize] [0,1,2, ... , W-blocksize] => (0,0),(0,1)..(0,W-blocksize),(1,0),...,(H-blocksize,W-blocksize)
		for r in range(len(img8)):
			# if (mask[r][i:i + blocksize, j:j + blocksize] == 1).all():
			rmsVal = ((img8[r][i:i+overlap, j:j+blocksize] - refBlock[-overlap:, :])**2).mean()	# (이웃 블록의 오버랩 부분 - 각 블록의 오버랩 부분) 제곱 의 평균
			if rmsVal > 0:
				errMat.append([i,j,r,rmsVal])

	# errMat_2dLow = list(zip(*errMat))

	# minVal = np.min(errMat_2dLow[3])  # 에러범위 값 중 가장 작은 것

	errMat.sort(key=lambda x:x[3])	# err 작은것부터 오름차순 정렬

	errIndex = []
	errIndex.append(errMat[:5])
	errIndex = sum(errIndex, [])	# errIndex 에서 [] 한꺼풀 벗겨줌
	# for i in range(len(errMat)):
	# 	if errMat[i][3] < (1.0 + tolerance) * (minVal):
	# 		errIndex.append(errMat[i])

	# while (True):
	c = np.random.randint(len(errIndex))  # random.randint() : [최소값, 최대값) 랜덤 정수 / 0~len(y) 전까지 / len(y) == len(x)
	y, x, r = errIndex[c][0], errIndex[c][1], errIndex[c][2]
		# if (mask[r][y:y + blocksize, x:x + blocksize] == 1).all():
		# 	break

	return img8[r][y:y+blocksize, x:x+blocksize]	# 텍스쳐에서 해당 블록 return

# 기존 회전예제 이미지 코드
def old_generateTextureMap(image, blocksize, overlap, outH, outW, tolerance):	# main.py에서 사용되는 메인. tolerance : 허용요차
	print(">>fin_generateTextureMap")
	# 사용: generateTextureMap(image, block_size, overlap, outH, outW, args.tolerance)
	# ceil() : 소수점 자리의 숫자를 무조건 올리는 함수
	nH = int(ceil((outH - blocksize)*1.0/(blocksize - overlap)))	# 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?
	nW = int(ceil((outW - blocksize)*1.0/(blocksize - overlap)))	# 최종 이미지 크기에 오버랩 부분을 제외한 실제 블록들이 몇개 들어가는가?

	textureMap = np.zeros(((blocksize + nH*(blocksize - overlap)), (blocksize + nW*(blocksize - overlap)), image.shape[2]))
	# [(H기준 : nH(들어가는 블록개수) * (오버랩 뺀 블록실제사이즈) + 마지막에 오버랩 안되므로 블록 하나 더 사이즈) , (W기준 동일) , 색상] => 0으로 초기화
	# Starting index and block
	H, W = image.shape[:2]
	pre_img8 = old_RotateExImg(image)	# pre_img8 : [ [rotated_seta , mask] , [rotated_seta , mask] , .. ]
																		# => shape : (8, 2, h, w, 3)
	tmp_img8 = list(zip(*pre_img8))	# [ [rotated_seta 끼리 ] , [mask 끼리] ] 로 형태 변환
	img8 = tmp_img8[0]
	mask = tmp_img8[1]

	randH = np.random.randint(H - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값
	randW = np.random.randint(W - blocksize)  # 블록사이즈 한줄 뺀 값에서 랜덤한 값

	startBlock = image[randH:randH+blocksize, randW:randW+blocksize]	# 랜덤한 위치에서 시작하는 블록 사이즈만큼 잘라서 가져옴
	textureMap[:blocksize, :blocksize, :] = startBlock	# 0으로 초기화된 맵에서 첫번째 블록에 랜덤하게 가져온 블록 대입함

	print(">>r_generateTextureMap 고고싱")
	# Fill the first row : 행(아래 위)
	for i, blkIdx in enumerate(range((blocksize-overlap), textureMap.shape[1]-overlap, (blocksize-overlap))):	# enumerate() : 인덱스와 원소 차례로 반환
		# 오버랩 부분 제외 블록 부분부터 ~ 오버랩 제외 열들까지 , 오버랩 제외한 블록사이즈만큼 옆으로 이동 (오른 -> 왼)

		# Find horizontal error for this block
		# Calculate min, find index having tolerance
		# Choose one randomly among them
		# blkIdx = block index to put in
		# blkIdx = 블록에서 오버랩 되는 부분 시작점 인덱스
		refBlock = textureMap[:blocksize, (blkIdx-blocksize+overlap):(blkIdx+overlap)]	#texturemap 의 한줄제외 모든 행에 대하여 열단위로 블록 한 칸만큼 계속 이동하면서 대입
		patchBlock = old_findPatchHorizontal(refBlock, img8, blocksize, overlap, tolerance, mask)	# 미리 만든 패치 찾는 함수
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
		patchBlock = old_findPatchVertical(refBlock, img8, blocksize, overlap, tolerance, mask)	# 미리 만든 패치 찾는 함수
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

			patchBlock = old_findPatchBoth(refBlockLeft, refBlockTop, img8, blocksize, overlap, tolerance, mask)
			minCutPatch = getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap)

			textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ):(blkIndexJ+blocksize)] = minCutPatch

			# refBlockLeft = 0.5
			# textureMap[(blkIndexI):(blkIndexI+blocksize), (blkIndexJ-blocksize+overlap):(blkIndexJ+overlap)] = refBlockLeft
			# textureMap[(blkIndexI-blocksize+overlap):(blkIndexI+overlap), (blkIndexJ):(blkIndexJ+blocksize)] = [0.5, 0.6, 0.7]
			# break
		print("{} out of {} rows complete...".format(i+1, nH+1))

	return textureMap