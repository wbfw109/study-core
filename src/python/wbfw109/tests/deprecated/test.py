# %%
#
import copy
import pprint

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
pp = pprint.PrettyPrinter(compact=False)


#%%

#   """
#     + Formula
#         sample_weight = dividend / the number of image in a directory

#     Args:
#         input_directory_list (list[Union[Path, str]]): [description]
#         base_weight_mode (str, optional): [description]. Defaults to "min".
#         prints_object_class_size (bool, optional): [description]. Defaults to False.

#     Returns:
#         dict[str, float]: [description]
#   """
#     print(f"===== Start function; {inspect.currentframe().f_code.co_name}")

#     assert len(input_directory_list) == len(input_directory_ratio_list)

#     for i in range(len(input_directory_list)):
#         input_directory_list[i] = Path(input_directory_list[i])
#     input_directory_size_list = [
#         int(len(list(input_directory.iterdir())) / 2)
#         for input_directory in input_directory_list
#     ]

#     if base_weight_mode == "min":
#         dividend: int = min(input_directory_size_list)
#     elif base_weight_mode == "max":
#         dividend: int = max(input_directory_size_list)
#     elif base_weight_mode == "mean":
#         dividend: int = sum(input_directory_size_list) / len(input_directory_size_list)

#     sample_dataset_weight_dict: dict[int, float] = {}
#     for i in range(len(input_directory_list)):
#         sample_dataset_weight_dict[i] = dividend / input_directory_size_list[i]

#     if prints_dataset_size:
#         for i in range(len(input_directory_list)):
#             print(
#                 (
#                     input_directory_list[i],
#                     input_directory_size_list[i],
#                     sample_dataset_weight_dict[i],
#                     input_directory_size_list[i] * sample_dataset_weight_dict[i],
#                 )
#             )
#     return list(
#         zip(
#             [input_directory.name for input_directory in input_directory_list],
#             sample_dataset_weight_dict.values(),
#         )
#     )

# tf_object_detection_api_config_format = [
#     value
#     for value in get_sample_weight_from_dataset_directory_list_as_custom_ratio(
#         input_directory_list=[
#             Path(
#                 "/mnt/c/users/wbfw109/image_tasks_sorted_true/pickup_bar-image_and_pascal_voc_set-1"
#             ),
#             Path(
#                 "/mnt/c/users/wbfw109/image_tasks_sorted_true/pickup_bar-image_and_pascal_voc_set-2"
#             ),
#             Path(
#                 "/mnt/c/users/wbfw109/image_tasks_sorted_true/pickup_bar-image_and_pascal_voc_set-3"
#             ),
#             Path(
#                 "/mnt/c/users/wbfw109/image_tasks_sorted_true/pickup_bar-image_and_pascal_voc_set-4"
#             ),
#             Path(
#                 "/mnt/c/users/wbfw109/image_tasks_sorted_true/pickup_bar-image_and_pascal_voc_set-5"
#             ),
#             Path(
#                 "/mnt/c/users/wbfw109/image_tasks_sorted_true/pickup_bar-image_and_pascal_voc_set-6"
#             ),
#             Path("/mnt/c/users/wbfw109/image_task_211019_AreaSample_no_exif_true"),
#         ],
#         base_weight_mode="max",
#         prints_object_class_size=False,
#     )
# ]
# pp.pprint(tf_object_detection_api_config_format)
#%%

# object_classes = [
#     "mug",
#     "takeout_hot",
#     "takeout_ice",
#     "tumbler",
#     "bagel",
#     "cheese_cake",
#     "croissant",
#     "fruit_cake",
#     "macaron",
#     "muffin",
#     "scone",
#     "square_sandwich",
#     "tiramisu",
#     "triangle_sandwich",
# ]

# for object_class in object_classes:
#     TEST_IMAGE_FILE_LIST: list[Path] = sorted(
#         list(test_image_directory.glob(f"*{object_class}*.jpg"))
#     )
#     for c, image_file in enumerate(TEST_IMAGE_FILE_LIST, start=1):
#         print(f"image_path: {image_file}")
#         show_inference(detection_model, image_file, inference_result_directory, creates_pascal_voc_file=False)
#         print(f"{c} done for {object_class}")
#         print("================\n")
#     break

# from tensorflow.python.framework.ops import EagerTensor

#%%

# 인덱스 2이 이상이고 비율 리스트랑 path 리스트 개수 같은지. 검증하고 시작, path 안의 파일개수도 모두 1 이상이어야 함.
c_temp_number_of_files_list = [100, 350]
c_propotion_list_to_want = [1.0, 3.0]
base_weight_unit_index = c_propotion_list_to_want.index(min(c_propotion_list_to_want))

# 기준 유닛의 coefficient 를 1. 으로 가정.  는 파일 개수.. 아 결국 인덱스는 유지해야되서 잘못됨.. 그냥 continue 하자 뒤에서 인덱스면.
c_index_to_sample_weight_dict: dict[int, float] = {
    c_temp_number_of_files_list[base_weight_unit_index]: 1.0
}

c_temp_number_of_files_list_except_base_index = copy.deepcopy(
    c_temp_number_of_files_list
)
c_propotion_to_want_except_base_index = copy.deepcopy(c_propotion_list_to_want)

# 0 ~ last, base_weight_unit_index 무시하면서? 순회할 리스트에서 자기꺼 뺴고
for i in range(len(c_propotion_list_to_want)):
    if i == base_weight_unit_index:
        continue

    c_index_to_sample_weight_dict[c_temp_number_of_files_list[i]] = round(
        c_temp_number_of_files_list[base_weight_unit_index]
        * c_propotion_list_to_want[i]
        / (
            c_temp_number_of_files_list[i]
            * c_propotion_list_to_want[base_weight_unit_index]
        ),
        2,
    )

c_index_to_sample_weight_dict
c_propotion_list_to_want

#%%
"""
a*파일개수: b*파일개수 = 1: 2
b*파일개수B*1 = 2*a*파일개수A
b = 파일개수A*원하는 B비율/ (파일개수b*원하는 a비율)

a : b : c = 1: 2: 3
b : c = 2: 4



a: b: c = 1: 2: 4
c = 2b = 4a

for input_directory_list

4a - 2b = 0
2b - c = 0

그냥 차례대로 계산하는수밖ㅇ ㅔ없나..

+ Formula
    a sample_weight * <the number of image in "A" directory> : another sample_weight * <the number of image in "B" directory>
        = propotion you want to of "B" dataset  :  propotion you want to of "B" dataset

... if 많다면..

Find ratio A:B:C:D Strategy to solve Ratio Word Problems


    sample_weight = <propotion you want to> / <the number of image in a directory>
    50*sample_weight*2 = 100*anohter_sampleweight*1
    50, 100
    1 : 2



    a b   x  = 
c d   y

raw_detection_boxes
detection_scores
detection_boxes
raw_detection_scores
detection_classes
detection_multiclass_scores
num_detections


result = math.product(이미지 파일, (배경합성 파일 포함) * augmentation * xml 파일)
mug: 2016
    18, 2, 28 * 2
cheese_cake: 3024
    27, 2, 28 * 2

## batch size.. 클래스당 18*2*28 = min 1008 장 ~ max 27*2*28 = max 1512 장
    batch_size 는 매 루프마다 각 클래스의 개수만큼 넣자.

클래스별 개수가 달라서 sample_from_datasets_weights 설정 필요할듯?

https://stackoverflow.com/questions/54561558/can-tensorflow-shuffle-multiple-sharded-tfrecord-binaries-for-object-detection-t

filenames_shuffle_buffer_size in input_reader.proto  vs  shuffle_buffer_size in input_reader.proto
    filenames_shuffle_buffer_size in input_reader.proto 
        Buffer size to be used when shuffling file names. I set this as max of the number of each object class. 27*2*28
        하나의 tfrecord 안에서 섞이는 이미지 파일의 버퍼사이즈를 말하는듯.
    shuffle_buffer_size in input_reader.proto
        Buffer size to be used when shuffling.
        tfrecord 파일들 사이에 섞이는 버퍼사이즈를 말하는듯.

num_readers in input_reader.proto
    Number of file shards to read in parallel. When sample_from_datasets_weights are configured, num_readers is applied for each dataset.
    이 값* 각각의 same_from_datasets_weights 만큼의 수가 곱해져서 그만큼을 샘플링하는듯.

num_epochs in input_reader.proto  vs  num_steps train.proto
    epoch 수를 기준으로 train 할 것인지, steps 를 기준으로 train 할 것인지 설정할 수 있는듯.
    = 이미지가 많아지면 step 을 기준으로 하여 학습을 중간에 멈추거나 재개하는 것이 더 도움이 될 것 같다.

batch_size in train.proto
    Effective batch size to use for training. For TPU (or sync SGD jobs), the batch size per core (or GPU) is going to be `batch_size` / number of cores (or `batch_size` / number of GPUs).
        추가적으로 최소 클래수 개수만큼 하는 것이 낫지 않나? SGD 는 배치 사이즈 한번마다 loss 가 바뀌는데 클래스별 균일하게 뽑는 것이 유리할지도? 32

num_readers in input_reader.proto, sample_from_datasets_weights in input_reader.proto
    num_readers in input_reader.proto
        각 tfrecord 로부터 sample_from_datasets_weights 을 곱해서 해당 개수만큼 뽑아서 이들 중 batch_size buffer 에 차례대로 채워지는 것으로 추정.
    sample_from_datasets_weights in input_reader.proto
        ..
        최소 클래스별 개수, 최대 클래수별 개수 비율이 2 : 3 이므로 동일하게 맞춰주기 위해 0.66 : 1.00 으로 하자.
    https://www.tensorflow.org/api_docs/python/tf/data/experimental/sample_from_datasets

    input_path: "/content/drive/MyDrive/shared_resource/annotations/train_*.record"
"""
# %%
