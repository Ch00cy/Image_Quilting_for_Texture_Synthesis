import numpy as np	# 벡터 및 행렬 연산
import  argparse	# 호출 당시 인자값을 줘서 동작을 다르게 하고 싶은 경우
import cv2	# OpenCV import -> 오픈소스 컴퓨터 비전 및 머신러닝 라이브러리
			# 먼저 설치 : pip install opencv-python
from matplotlib import pyplot as plt	# matplotlib : 자료를 시각화 하는데 사용하는 대표 라이브러리 , 그래프 등 그림
from utils.preprocess import *	# utils 파일에 preprocess.py 의 모든 내용 가져오기
from utils.generate import *	# utils 파일에 generate.py 의 모든 내용 가져오기
from math import ceil	# math 함수 -> ceil : 올림 (int형)

## Get parser arguments	(호출당시 인자값 -> 동작)
parser = argparse.ArgumentParser()	# ArgumentParser 객체

# - : 인수명이 한 문자인 경우에는 1개, 2개 이상의 문자수를 가진 인수명인 경우는 2개
# required=True : 필수 지정 옵션 인수
# type : 데이터타입 지정
# default : 옵션 인수 지정 X 경우 -> None 아닌 디폴트값 설정
# help : 설명
parser.add_argument("-i", "--image_path", required=True, type=str, help="path of image you want to quilt")	# 이미지 경로 - 필수
parser.add_argument("-b", "--block_size", type=int, default=20, help="block size in pixels")	# 블록 사이즈 (20픽셀)
parser.add_argument("-o", "--overlap", type=int, default=1.0/6, help="overlap size in pixels (defaults to 1/6th of block size)")	# 오버랩 부분 (1/6 픽셀)
parser.add_argument("-s", "--scale", type=float, default=4, help="Scaling w.r.t. to image size")	# 결과 이미지 사이즈 얼마나 배로 늘릴것인가 (4)
parser.add_argument("-n", "--num_outputs", type=int, default=1, help="number of output textures required")	# 결과 텍스쳐 몇개 생성? (1)
parser.add_argument("-f", "--output_file", type=str, default="output.png", help="output file name")	# 결과 어디에 저장? (output.png)
parser.add_argument("-p", "--plot", type=int, default=1, help="Show plots")	# plot 보여줄 여부 (1)
parser.add_argument("-t", "--tolerance", type=float, default=0.1, help="Tolerance fraction")	# 허용오차 (0.1)

args = parser.parse_args()	# 해당 메서드를 통해 인자 파싱(파싱 : 어떤 큰 자료에서. 내가 원하는 정보만 가공하고 추출해서. 원할 때 불러올 수 있게 하는 것)

if __name__ == "__main__":	# 해당 main.py 가 메인으로 불려왔을 때 실행
	# Start the main loop here
	path = args.image_path	# (호출인자)이미지 경로 - 사용자가 입력
	block_size = args.block_size	# (호출인자)블록 사이즈 - 디폴트 20픽셀
	scale = args.scale	# (호출인자)결과 이미지 사이즈 얼마나 배로 늘릴것인가 - 디폴트 4
	overlap = args.overlap	# (호출인자)오버랩 부분 - 디폴트 1/6
	print("Using plot {}".format(args.plot))	# plot(그래프, 시각화???) - 디폴트 1
	# Set overlap to 1/6th of block size

	if overlap > 0:	# 디폴트 overlap = 1/6
		overlap = int(block_size*args.overlap)	# 블록 사이즈의 1/6
	else:
		overlap = int(block_size/6.0)

	# Get all blocks
	image = cv2.imread(path)	# 이미지 읽어오기
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0	# openCv : 컬러값 BGR -> RGB 변경 => 0~255 -> 0~1 값 변경
	print("Image size: ({}, {})".format(*image.shape[:2]))
	print("Image shape: {}".format(image.shape))

	image = cv2.resize(image, (200, 200))
	H, W = image.shape[:2]	# 이미지 Height, Width
	outH, outW = int(scale*H), int(scale*W)	# 아웃풋 결과 : 이미지의 scale(4) 배로 키워줌

	# 수정 - 추가부분
	# 이미지 사이즈 w,h중 작은 것에 맞춰서 정사각형으로 크기조절
	# if H>W:
	# 	image = cv2.resize(image, (W, W))
	# else:
	# 	image = cv2.resize(image, (H, H))

	for i in range(args.num_outputs):	# 결과 개수 만큼 반복
		# 수정
		#textureMap = r_generateTextureMap(image, block_size, overlap, H, W, args.tolerance)
		textureMap = multi_RotateExImg(image, block_size, overlap, outH, outW, args.tolerance)

		# textureMaps = Pre_RotateExImg(image, image, block_size, overlap, outH, outW, args.tolerance)
		# textureMaps = Pre_AddRotateIndex(textureMaps)
		# textureMaps = Pre_FindNeighbor(textureMaps)

		# Save
		# textureMap = (255*textureMap).astype(np.uint8)	# 최종 결과 텍스쳐 맵 -> 0~1, RGB 형태 => 원래대로로 돌림 (0~155 , BGR형태 , unit8)
		# textureMap = cv2.cvtColor(textureMap, cv2.COLOR_RGB2BGR)
		#
		# cv2.imwrite(args.output_file, textureMap)

		# #textureMap = generateTextureMap(image, block_size, overlap, outH, outW, args.tolerance)	# generate.py -> generateTextureMap(image, blocksize, overlap, outH, outW, tolerance) 함수 실행
		# if args.plot:	# plot 보여줄지 true 면 실행 (디폴트 1)
		# 	plt.imshow(textureMap)	# array의 값들을 색으로 환산해 이미지의 형태로 보여줌
		# 	plt.show()	# array의 값들을 색으로 환산해 이미지의 형태로 보여줌



		# #Save
		# textureMap = (255*textureMap).astype(np.uint8)	# 최종 결과 텍스쳐 맵 -> 0~1, RGB 형태 => 원래대로로 돌림 (0~155 , BGR형태 , unit8)
		# textureMap = cv2.cvtColor(textureMap, cv2.COLOR_RGB2BGR)
		#
		# if args.num_outputs == 1:
		# 	cv2.imwrite("output.png", textureMap)
		# 	print("Saved output to {}".format(args.output_file))
		# else:
		# 	cv2.imwrite(args.output_file.replace(".", "_{}.".format(i)), textureMap)
		# 	print("Saved output to {}".format(args.output_file.replace(".", "_{}.".format(i))))