
###### 임포트
import pandas as pd
import numpy as np
import joblib
import json

###### 로드 및 추출
df = pd.read_pickle(rf'C:/Users/401-14/Desktop/30-200/mel_genre/extracted_dataframe.pickle') # CNN 특징추출 (장르)
loaded_model = joblib.load(rf'C:/Users/401-14/Desktop/30-200/mel_genre/knn_model.joblib')    # KNN 이웃추천
X = np.array( [np.array(x) for x in df['total']] )           # Data 정렬

###### 함수정의

# song_id -> index
def get_indices_from_song_ids(song_ids):
    indices = []
    for song_id in song_ids:
        try:
            index = df[df['key'] == song_id].index.values[0]
            indices.append(index)
        except IndexError:
            pass
    return indices

# find random neighbor indices (recommend neighbor caculated by mel_spectrum with genre)
def get_random_neighbor_indices(index_list, k=10, random_samples=5):
    random_neighbors = set()
    for index in index_list:
        query_point = X[index].reshape(1, -1)
        indices = loaded_model.kneighbors(query_point, n_neighbors=k, return_distance=False)
        random_neighbors.update(indices[0][1:])
    
    # Convert the set to a list and shuffle
    random_neighbors_list = list(random_neighbors)
    np.random.shuffle(random_neighbors_list)
    
    # Take random samples (default: 5)
    random_neighbors_list = random_neighbors_list[:random_samples]
    
    return random_neighbors_list

# convert list of indices to song_id values
def indices_to_song_id(data_frame, indices_list):
    song_id_list = [data_frame.loc[index, 'key'] for index in indices_list]
    return song_id_list

def get_data_by_keys(df, random_neighbors):
    data_list = []
    for index, row in df.iterrows():
        # Check if the 'key' column value is
        if row['key'] in random_neighbors:
            # Create a dictionary
            data_dict = {
                'key': row['key'], ##### 만약에 key를 song_id로 전달해줘야하면 수정해야할 부분.
                'song_name': row['song_name'],
                'artist_name': row['artist_name'],
                'lyric_str': row['lyric_str'],
                'url_new_song': row['url_new_song']
            }

            # Add the dictionary to the list
            data_list.append(data_dict)
            # Convert the list of dictionaries to a JSON string
            json_data = json.dumps(data_list)
            
    return data_list ##### json_data(json_dump를 해야할 경우)

# 전체 순차 실행 함수
def run(index_list):
    input_to_function = get_indices_from_song_ids(index_list)
    # print("Input to Function:", input_to_function)
    random_neighbors = get_random_neighbor_indices(input_to_function, k=10)
    # print("Random Neighbor Indices:", random_neighbors)
    song_id_list = indices_to_song_id(df, random_neighbors)
    print("Random Neighbor Song IDs:", song_id_list) ##### !!!!! 이 부분에 song_id_list가 출력되는 게 아니라, song_id_list를 DB에 보내지게 바꿔줘야함.
    result = get_data_by_keys(df, song_id_list)
    return result

###### 실행부분
input_list = [6903, 8446, 3025, 801, 1998] ##### 인풋 : 임의로 지정했지만, 스프링부트에서 들어오는 인풋리스트로 변환. (input value - integer list --- "KEY")
output_json = run(input_list) ##### 아웃풋 :  type | list[dict] -> "json"형식으로 바꿔야 하면 변경해야함. (json.dump로 쏘면 될 것 같긴함.)
# print("Output:", output_json) ##### 확인용