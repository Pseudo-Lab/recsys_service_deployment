import pandas as pd
import pickle
import argparse
from collections import defaultdict
import random
from tqdm import tqdm
import re

import sys
from os import path, mkdir

sys.path.append(path.dirname(path.dirname(path.abspath('./constants'))))
import constants.consts as consts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--movies_file',
                        default='movies.csv',
                        help='Path to the CSV file containing movie information')
    parser.add_argument('--interactions_file',
                        default='train.csv',
                        help='Path to the CSV file containing user movie interactions')
    parser.add_argument('--subnetwork',
                        default='dense',
                        choices=['dense', 'rs', 'sparse', 'full'],
                        help='The type of subnetwork to form from the full KG')
    return parser.parse_args()

def get_personId(person_link):
    try:
        str_idx = person_link.find('personId=')
        return person_link[str_idx+9:]
    except:
        return ''

def make_person_list(row):
    try:
        person_set = re.findall(r'personId=(\d+)', row['cast'])
        return list(set(person_set))
    except TypeError:
        return []

def movie_data_prep(movies_file, interactions_file, export_dir):
    '''
    :return: Write out 4 python dictionaries for the edges of KG
    '''

    movies = pd.read_csv(movies_file)
    interactions = pd.read_csv(interactions_file)

    # movie_person.dict
    # dict where key = movieId, value = list of persons (감독, 주연, 출연) of the movie
    # 수정 완료
    person = movies[['movieId', 'cast']]
    person_list = person.apply(lambda x: make_person_list(x), axis=1)
    movie_person = pd.concat([movies['movieId'], person_list], axis=1)
    movie_person.columns = ['movieId', 'personIds']
    movie_person_dict = movie_person.set_index('movieId')['personIds'].to_dict()
    with open(export_dir + consts.MOVIE_PERSON_DICT, 'wb') as handle:
        pickle.dump(movie_person_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # person_movie.dict
    # dict where key = a person, value = list of movies related to this person
    # 위에 것 수정 끝나면 건드릴 필요 없음
    person_movie_dict = {}
    for row in movie_person.iterrows():
        for person in row[1]['personIds']:
            if person not in person_movie_dict:
                person_movie_dict[person] = []
            person_movie_dict[person].append(row[1]['movieId'])
    with open(export_dir + consts.PERSON_MOVIE_DICT, 'wb') as handle:
        pickle.dump(person_movie_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # movie_user.dict
    # dict where key = movie_id, value = list of user_ids
    # 수정 완료
    movie_user = interactions[['movieId', 'nickName']].set_index('movieId').groupby('movieId')['nickName'].apply(list).to_dict()
    # msno is the user_id
    with open(export_dir + consts.MOVIE_USER_DICT, 'wb') as handle:
        pickle.dump(movie_user, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # user_movie.dict
    # dict where key = user_id, value = list of movie_ids
    # 위에 것 수정 끝나면 건드릴 필요 없음
    user_movie = interactions[['nickName', 'movieId']].set_index('nickName').groupby('nickName')['movieId'].apply(list).to_dict()
    with open(export_dir + consts.USER_MOVIE_DICT, 'wb') as handle:
        pickle.dump(user_movie, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # user_movie_tuple.txt
    # numpy array of [user_id, movie_id] pairs sorted in the order of user_id
    # 수정 완료?
    user_movie_tuple = interactions[['nickName', 'movieId']].sort_values(by='nickName').to_string(header=False, index=False,
                                                                                         index_names=False).split('\n')
    user_movie_tuple = [row.split() for row in user_movie_tuple]
    with open('user_movie_tuple.txt', 'wb') as handle:
        pickle.dump(user_movie_tuple, handle, protocol=pickle.HIGHEST_PROTOCOL)


def find_subnetwork(network_type, dir, factor=0.1):
    if network_type == 'full':
        return

    # Load Data

    with open(dir + consts.MOVIE_USER_DICT, 'rb') as handle:
        movie_user = pickle.load(handle)
    with open(dir + consts.USER_MOVIE_DICT, 'rb') as handle:
        user_movie = pickle.load(handle)
    with open(dir + consts.MOVIE_PERSON_DICT, 'rb') as handle:
        movie_person = pickle.load(handle)
    with open(dir + consts.PERSON_MOVIE_DICT, 'rb') as handle:
        person_movie = pickle.load(handle)
    movie_user = defaultdict(list, movie_user)
    movie_person = defaultdict(list, movie_person)
    user_movie = defaultdict(list, user_movie)
    person_movie = defaultdict(list, person_movie)

    # Sort Nodes By Degree in descending order

    # key: movie, value: number of users listening to it + number of person relating to its creation
    movie_degree_dict = {}
    for (k, v) in movie_user.items():
        movie_degree_dict[k] = v
    for (k, v) in movie_person.items():
        if k in movie_degree_dict.keys():
            movie_degree_dict[k] = movie_degree_dict[k] + v
        else:
            movie_degree_dict[k] = v
    movie_degree = [(k, len(v)) for (k, v) in movie_degree_dict.items()]
    movie_degree.sort(key=lambda x: -x[1])

    # key: person, value: number of movies they participated
    person_degree = [(k, len(v)) for (k, v) in person_movie.items()]
    person_degree.sort(key=lambda x: -x[1])

    # key: user, value: number of movies they watched
    user_degree = [(k, len(v)) for (k, v) in user_movie.items()]
    user_degree.sort(key=lambda x: -x[1])

    # Construct Subnetworks

    # find the nodes
    print('finding the nodes...')
    if network_type == 'dense':
        movie_nodes_holder = movie_degree[:int(len(
            movie_degree) * factor)]  # movie_id is the first item in the tuple element of the returned list
        movie_nodes = [node_holder[0] for node_holder in movie_nodes_holder]

        user_nodes_holder = user_degree[:int(len(user_degree) * factor)]
        user_nodes = [node_holder[0] for node_holder in user_nodes_holder]

        person_nodes_holder = person_degree[:int(len(person_degree) * factor)]
        person_nodes = [node_holder[0] for node_holder in person_nodes_holder]

    # random sampling
    elif network_type == 'rs':
        movie_nodes_holder = random.sample(movie_degree, int(len(
            movie_degree) * factor))  # movie_id is the first item in the tuple element of the returned list
        movie_nodes = [node_holder[0] for node_holder in movie_nodes_holder]

        user_nodes_holder = random.sample(user_degree, int(len(user_degree) * factor))
        user_nodes = [node_holder[0] for node_holder in user_nodes_holder]

        person_nodes_holder = random.sample(person_degree, int(len(person_degree) * factor))
        person_nodes = [node_holder[0] for node_holder in person_nodes_holder]

    elif network_type == 'sparse':
        movie_nodes_holder = movie_degree[-int(len(
            movie_degree) * factor):]  # movie_id is the first item in the tuple element of the returned list
        movie_nodes = [node_holder[0] for node_holder in movie_nodes_holder]

        user_nodes_holder = user_degree[-int(len(user_degree) * factor):]
        user_nodes = [node_holder[0] for node_holder in user_nodes_holder]

        person_nodes_holder = person_degree[-int(len(person_degree) * factor):]
        person_nodes = [node_holder[0] for node_holder in person_nodes_holder]

    nodes = movie_nodes + user_nodes + person_nodes
    print('The %s subnetwork has %d nodes: %d movies, %d users, %d persons.' % (network_type, \
                                                                               len(nodes), \
                                                                               len(movie_nodes), \
                                                                               len(user_nodes), \
                                                                               len(person_nodes)))
    # find the edges
    # (node1, node2) and (node2, node1) both exist
    edges_type1 = []  # a list of pairs (movie, user)
    edges_type2 = []  # a list of pairs (movie, person)
    edges_type3 = []  # a list of pairs (user, movie)
    edges_type4 = []  # a list of pairs (person, movie)
    nodes_set = set(nodes)

    for i in tqdm(nodes_set):  # (node1, node2) and (node2, node1) both exist
        connect_1 = set(movie_user[i]).intersection(nodes_set)
        for j in connect_1:
            edges_type1.append((i, j))

        connect_2 = set(movie_person[i]).intersection(nodes_set)
        for j in connect_2:
            edges_type2.append((i, j))

        connect_3 = set(user_movie[i]).intersection(nodes_set)
        for j in connect_3:
            edges_type3.append((i, j))

        connect_4 = set(person_movie[i]).intersection(nodes_set)
        for j in connect_4:
            edges_type4.append((i, j))

    edges = edges_type1 + edges_type2 + edges_type3 + edges_type4
    print('The %s subnetwork has %d edges.' % (network_type, len(edges)))

    # Export the Subnetworks

    # <NETWORK_TYPE>_movie_user.dict
    # key: movie, value: a list of users
    movie_user_dict = defaultdict(list)
    for edge in edges_type1:
        movie = edge[0]
        user = edge[1]
        movie_user_dict[movie].append(user)
    movie_user_dict = dict(movie_user_dict)
    prefix = dir + network_type
    with open(prefix + consts.MOVIE_USER_DICT, 'wb') as handle:
        pickle.dump(movie_user_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # <NETWORK TYPE>_movie_person.dict
    # key: movie, value: a list of persons
    movie_person_dict = defaultdict(list)
    for edge in edges_type2:
        movie = edge[0]
        person = edge[1]
        movie_person_dict[movie].append(person)
    movie_person_dict = dict(movie_person_dict)
    with open(prefix + consts.MOVIE_PERSON_DICT, 'wb') as handle:
        pickle.dump(movie_person_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # <NETWORK TYPE>_user_movie.dict
    # key: user, value: a list of movies
    user_movie_dict = defaultdict(list)
    for edge in edges_type3:
        user = edge[0]
        movie = edge[1]
        user_movie_dict[user].append(movie)
    user_movie_dict = dict(user_movie_dict)
    with open(prefix + consts.USER_MOVIE_DICT, 'wb') as handle:
        pickle.dump(user_movie_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # <NETWORK TYPE>_person_movie.dict
    # key: person, value: a list of movies
    person_movie_dict = defaultdict(list)
    for edge in edges_type4:
        person = edge[0]
        movie = edge[1]
        person_movie_dict[person].append(movie)
    person_movie_dict = dict(person_movie_dict)
    with open(prefix + consts.PERSON_MOVIE_DICT, 'wb') as handle:
        pickle.dump(person_movie_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_to_ids(entity_to_ix, rel_dict, start_type, end_type):
    new_rel = {}
    for key, values in rel_dict.items():
        key_id = entity_to_ix[(key, start_type)]
        value_ids = []
        for val in values:
            try:
                value_ids.append(entity_to_ix[(val, end_type)])
            except KeyError:
                print(key)
                print(values)
        new_rel[key_id] = value_ids
    return new_rel

def ix_mapping(network_type, import_dir, export_dir, mapping_export_dir):
    pad_token = consts.PAD_TOKEN
    type_to_ix = {'person': consts.PERSON_TYPE, 'user': consts.USER_TYPE, 'movie': consts.MOVIE_TYPE,
                  pad_token: consts.PAD_TYPE}
    relation_to_ix = {'movie_person': consts.MOVIE_PERSON_REL, 'person_movie': consts.PERSON_MOVIE_REL,
                      'user_movie': consts.USER_MOVIE_REL, 'movie_user': consts.MOVIE_USER_REL, '#UNK_RELATION': consts.UNK_REL,
                      '#END_RELATION': consts.END_REL, pad_token: consts.PAD_REL}

    # entity vocab set is combination of movies, users, and persons
    movie_data_prefix = import_dir + network_type
    with open(movie_data_prefix + consts.MOVIE_USER_DICT, 'rb') as handle:
        movie_user = pickle.load(handle)
    with open(movie_data_prefix + consts.MOVIE_PERSON_DICT, 'rb') as handle:
        movie_person = pickle.load(handle)
    with open(movie_data_prefix + consts.USER_MOVIE_DICT, 'rb') as handle:
        user_movie = pickle.load(handle)
    with open(movie_data_prefix + consts.PERSON_MOVIE_DICT, 'rb') as handle:
        person_movie = pickle.load(handle)

    movies = set(movie_user.keys()) | set(movie_person.keys())
    users = set(user_movie.keys())
    persons = set(person_movie.keys())

    # Id-ix mappings
    entity_to_ix = {(movie, consts.MOVIE_TYPE): ix for ix, movie in enumerate(movies)}
    entity_to_ix.update({(user, consts.USER_TYPE): ix + len(movies) for ix, user in enumerate(users)})
    entity_to_ix.update(
        {(person, consts.PERSON_TYPE): ix + len(movies) + len(users) for ix, person in enumerate(persons)})
    entity_to_ix[pad_token] = len(entity_to_ix)

    # Ix-id mappings
    ix_to_type = {v: k for k, v in type_to_ix.items()}
    ix_to_relation = {v: k for k, v in relation_to_ix.items()}
    ix_to_entity = {v: k for k, v in entity_to_ix.items()}

    # Export mappings
    movie_ix_mapping_prefix = mapping_export_dir + network_type
    # eg. movie_ix_data/dense_type_to_ix.dict
    with open(movie_ix_mapping_prefix + consts.TYPE_TO_IX, 'wb') as handle:
        pickle.dump(type_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(movie_ix_mapping_prefix + consts.RELATION_TO_IX, 'wb') as handle:
        pickle.dump(relation_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(movie_ix_mapping_prefix + consts.ENTITY_TO_IX, 'wb') as handle:
        pickle.dump(entity_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(movie_ix_mapping_prefix + consts.IX_TO_TYPE, 'wb') as handle:
        pickle.dump(ix_to_type, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(movie_ix_mapping_prefix + consts.IX_TO_RELATION, 'wb') as handle:
        pickle.dump(ix_to_relation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(movie_ix_mapping_prefix + consts.IX_TO_ENTITY, 'wb') as handle:
        pickle.dump(ix_to_entity, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Update the KG
    movie_user_ix = convert_to_ids(entity_to_ix, movie_user, consts.MOVIE_TYPE, consts.USER_TYPE)
    user_movie_ix = convert_to_ids(entity_to_ix, user_movie, consts.USER_TYPE, consts.MOVIE_TYPE)
    movie_person_ix = convert_to_ids(entity_to_ix, movie_person, consts.MOVIE_TYPE, consts.PERSON_TYPE)
    person_movie_ix = convert_to_ids(entity_to_ix, person_movie, consts.PERSON_TYPE, consts.MOVIE_TYPE)

    # export
    # eg. movie_ix_data/dense_ix_movie_user.dict
    ix_prefix = export_dir + network_type + '_ix_'
    with open(ix_prefix + consts.MOVIE_USER_DICT, 'wb') as handle:
        pickle.dump(movie_user_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix_prefix + consts.USER_MOVIE_DICT, 'wb') as handle:
        pickle.dump(user_movie_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix_prefix + consts.MOVIE_PERSON_DICT, 'wb') as handle:
        pickle.dump(movie_person_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix_prefix + consts.PERSON_MOVIE_DICT, 'wb') as handle:
        pickle.dump(person_movie_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_test_split(network_type, dir):
    with open(dir + network_type + '_ix_' + consts.USER_MOVIE_DICT, 'rb') as handle:
        user_movie = pickle.load(handle)

    # KG and positive
    train_user_movie = {}
    test_user_movie = {}
    train_movie_user = defaultdict(list)
    test_movie_user = defaultdict(list)

    for user in user_movie:
        pos_movies = user_movie[user]
        random.shuffle(pos_movies)
        cut = int(len(pos_movies) * 0.8)

        # train
        train_user_movie[user] = pos_movies[:cut]
        for movie in pos_movies[:cut]:
            train_movie_user[movie].append(user)

        # test
        test_user_movie[user] = pos_movies[cut:]
        for movie in pos_movies[cut:]:
            test_movie_user[movie].append(user)

    # Export
    # eg. movie_ix_data/dense_train_ix_movie_user.dict
    with open('%s%s_train_ix_%s' % (dir, network_type, consts.USER_MOVIE_DICT), 'wb') as handle:
        pickle.dump(train_user_movie, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s%s_test_ix_%s' % (dir, network_type, consts.USER_MOVIE_DICT), 'wb') as handle:
        pickle.dump(test_user_movie, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s%s_train_ix_%s' % (dir, network_type, consts.MOVIE_USER_DICT), 'wb') as handle:
        pickle.dump(train_movie_user, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s%s_test_ix_%s' % (dir, network_type, consts.MOVIE_USER_DICT), 'wb') as handle:
        pickle.dump(test_movie_user, handle, protocol=pickle.HIGHEST_PROTOCOL)

def create_directory(dir):
    print("Creating directory %s" % dir)
    try:
        mkdir(dir)
    except FileExistsError:
        print("Directory already exists")

def main():
    print("Data preparation:")
    args = parse_args()

    network_prefix = args.subnetwork
    if network_prefix == 'full':
       network_prefix = ''

    print("Forming knowledge graph...")
    create_directory(consts.MOVIE_DATA_DIR)
    movie_data_prep(consts.MOVIE_DATASET_DIR + args.movies_file,
                   consts.MOVIE_DATASET_DIR + args.interactions_file,
                   consts.MOVIE_DATA_DIR)

    print("Forming network...")
    find_subnetwork(args.subnetwork, consts.MOVIE_DATA_DIR)

    print("Mapping ids to indices...")
    create_directory(consts.MOVIE_IX_DATA_DIR)
    create_directory(consts.MOVIE_IX_MAPPING_DIR)
    ix_mapping(network_prefix, consts.MOVIE_DATA_DIR, consts.MOVIE_IX_DATA_DIR, consts.MOVIE_IX_MAPPING_DIR)

    print("Creating training and testing datasets...")
    train_test_split(network_prefix, consts.MOVIE_IX_DATA_DIR)

if __name__ == "__main__":
    main()
