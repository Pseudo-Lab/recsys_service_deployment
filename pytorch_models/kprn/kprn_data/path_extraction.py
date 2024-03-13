import sys
from os import path

import pickle
import random
from collections import defaultdict
import copy
import json

with open('pytorch_models/kprn/params.json', 'r') as f:
    params = json.load(f)
f.close()

class PathState:
    def __init__(self, path, length, entities):
        self.path = path    # array of [entity, entity type, relation to next] triplets
        self.length = length
        self.entities = entities    # set to keep track of the entities alr in the path to avoid cycles

def get_random_index(nums, max_length):
    index_list = list(range(max_length))
    random.shuffle(index_list)
    return index_list[:nums]


def find_paths_user_to_movies(start_user, movie_person, person_movie, movie_user, user_movie, max_length, sample_nums):
    '''
    Finds sampled paths of max depth from a user to a sampling of movies
    '''
    movie_to_paths = defaultdict(list)
    stack = []
    start = PathState([[start_user, params["USER_TYPE"], params["END_REL"]]], 0, {start_user})
    stack.append(start)
    while len(stack) > 0:
        front = stack.pop()
        entity, type = front.path[-1][0], front.path[-1][1]
        #add path to movie_to_user_paths dict, just want paths of max_length rn since either length 3 or 5
        if type == params["MOVIE_TYPE"] and front.length == max_length:
            movie_to_paths[entity].append(front.path)

        if front.length == max_length:
            continue

        if type == params["USER_TYPE"] and entity in user_movie:
            movie_list = user_movie[entity]
            index_list = get_random_index(sample_nums, len(movie_list))
            for index in index_list:
                movie = movie_list[index]
                if movie not in front.entities:
                    new_path = copy.deepcopy(front.path)
                    new_path[-1][2] = params["USER_MOVIE_REL"]
                    new_path.append([movie, params["MOVIE_TYPE"], params["END_REL"]])
                    new_state = PathState(new_path, front.length + 1, front.entities|{movie})
                    stack.append(new_state)

        elif type == params["MOVIE_TYPE"]:
            if entity in movie_user:
                user_list = movie_user[entity]
                index_list = get_random_index(sample_nums, len(user_list))
                for index in index_list:
                    user = user_list[index]
                    if user not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = params["MOVIE_USER_REL"]
                        new_path.append([user, params["USER_TYPE"], params["END_REL"]])
                        new_state = PathState(new_path, front.length + 1, front.entities|{user})
                        stack.append(new_state)
            if entity in movie_person:
                person_list = movie_person[entity]
                index_list = get_random_index(sample_nums, len(person_list))
                for index in index_list:
                    person = person_list[index]
                    if person not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = params["MOVIE_PERSON_REL"]
                        new_path.append([person, params["PERSON_TYPE"], params["END_REL"]])
                        new_state = PathState(new_path, front.length + 1, front.entities|{person})
                        stack.append(new_state)

        elif type == params["PERSON_TYPE"] and entity in person_movie:
            movie_list = person_movie[entity]
            index_list = get_random_index(sample_nums, len(movie_list))
            for index in index_list:
                movie = movie_list[index]
                if movie not in front.entities:
                    new_path = copy.deepcopy(front.path)
                    new_path[-1][2] = params["PERSON_MOVIE_REL"]
                    new_path.append([movie, params["MOVIE_TYPE"], params["END_REL"]])
                    new_state = PathState(new_path, front.length + 1, front.entities|{movie})
                    stack.append(new_state)

    return movie_to_paths


def main():
    with open("movie_data_ix/dense_ix_movie_person.dict", 'rb') as handle:
        movie_person = pickle.load(handle)

    with open("movie_data_ix/dense_ix_person_movie.dict", 'rb') as handle:
        person_movie = pickle.load(handle)

    with open("movie_data_ix/dense_ix_movie_user.dict", 'rb') as handle:
        movie_user = pickle.load(handle)

    with open("movie_data_ix/dense_ix_user_movie.dict", 'rb') as handle:
        user_movie = pickle.load(handle)

    print(find_paths_user_to_movies('빨간다라이', movie_person, person_movie, movie_user, user_movie, 3, 1))


if __name__ == "__main__":
    main()
