import json
import gzip
import sys
import os
import csv
import yaml
import os
import random

from datetime import datetime, timedelta
from bloc.util import getDictFromJson
from random import shuffle
from scipy.stats import wasserstein_distance
from collections import Counter
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from bloc.util import get_default_symbols
from bloc.generator import add_bloc_sequences

from info_ops_tk.util import get_bloc_lite_twt_frm_full_twt
from info_ops_tk.util import get_bloc_lite_twt
from info_ops_tk.util import parallelTask
from info_ops_tk.util import genericErrorInfo
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .utils import calculate_changes_for_all, segment_bloc_for_all, generate_bloc_for_all
from .classifier import classifier

name_mapper = {
    '2018_10/iranian' : 'Iran_1',
    '2019_01/bangladesh_201901_1' : 'Bangladesh',
    '2019_01/iran_201901_X/iran_201901_1' : 'Iran_2',
    '2019_01/russia_201901_1' : 'Russia_1',
    '2019_01/venezuela_201901_1' : 'Venezuela_1',
    '2019_01/venezuela_201901_2' : 'Venezuela_2',
    '2019_06/catalonia_201906_1' : 'Catalonia',
    '2019_06/iran_201906_1' : 'Iran_3',
    '2019_06/iran_201906_2' : 'Iran_4',
    '2019_06/iran_201906_3' : 'Iran_5',
    '2019_06/venezuela_201906_1' : 'Venezuela_3',
    '2019_08/china_082019_1' : 'China_1',
    '2019_08/china_082019_2' : 'China_2',
    '2019_08/ecuador_082019_1' : 'Ecuador',
    '2019_08/egypt_uae_082019_1' : 'Egypt_UAE',
    '2019_08/spain_082019_1' : 'Spain',
    '2019_08/uae_082019_1' : 'UAE',
    '2020_03/ghana_nigeria_032020' : 'Ghana_Nigeria',
    '2020_08/qatar_082020' : 'Qatar',
    '2020_09/ira_092020' : 'Russia_2',
    '2020_09/iran_092020' : 'Iran_6',
    '2020_09/thailand_092020' : 'Thailand',
    '2020_12/GRU_202012' : 'Russia_3',
    '2020_12/IRA_202012' : 'Russia_4',
    '2020_12/armenia_202012' : 'Armenia',
    '2020_12/iran_202012' : 'Iran_7',
    '2021_12/CNCC_0621_YYYY/CNCC_0621_2021' : 'China_3',
    '2021_12/CNHU_0621_YYYY/CNHU_0621_2020' : 'China_5',
    '2021_12/CNHU_0621_YYYY/CNHU_0621_2021' : 'China_4',
    '2021_12/MX_0621_YYYY/MX_0621_2019' : 'Mexico_1',
    '2021_12/Venezuela_0621_YYYY/Venezuela_0621_2020' : 'Venezuela_5',
    '2021_12/Venezuela_0621_YYYY/Venezuela_0621_2021' : 'Venezuela_4',
    '2021_12/uganda_0621_YYYY/uganda_0621_2019' : 'Uganda_1',
    '2021_12/uganda_0621_YYYY/uganda_0621_2020' : 'Uganda_2'
}

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def convert_to_ISO_time(date_str):
    return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")


def get_driver_control_per_day_tweets(filename):
    all_drivers = set()
    all_tweets = []

    driver_control_tweets = filename.replace('control_driver_users.csv', 'control_driver_tweets.jsonl.gz')
    with gzip.open(driver_control_tweets, 'rb') as f:
        for tweet in f:
            
            tweet = getDictFromJson(tweet)
            tweet = get_bloc_lite_twt_frm_full_twt(tweet)

            user_id = tweet['user']['id']

            all_drivers.add(user_id)
            
            all_tweets.append( tweet )
            
    return {
        'total_control': len(all_drivers),
        'control_driver_posts': all_tweets
    }

def get_driver_per_day_tweets(filename):
    
    if( os.path.exists(filename) is False ):
        return {}

    all_drivers = set()
    all_tweets = []

    with gzip.open(filename, 'rt') as infile:
        csvobj = csv.reader(infile)
        header = next(csvobj)
        hmap = {}
        [ hmap.update({ h[1]: h[0] }) for h in enumerate(header) ]
    
        while True:
            row = ''
            try:
                row = next(csvobj)
            except StopIteration:
                break
            except:
                genericErrorInfo()

            if( len(row) == 0 ):
                continue

            user_id = row[ hmap['userid'] ]

            tweet_time = row[ hmap['tweet_time'] ]

            if(not tweet_time):
                continue

            all_drivers.add(user_id)

            tweet = get_bloc_lite_twt(row, header)

            tweet['tweet_time'] = tweet_time
            all_tweets.append( tweet )

    return {
        'driver_posts': all_tweets,
        'total_drivers': len(all_drivers)
    }

def get_all_per_user_tweets(all_tweets, max_tweets, min_tweets):
    users_tweets = {}
    final_users_tweets = {}
    
    for twt in all_tweets:
        uid = twt['user']['id']
        users_tweets.setdefault(uid, [])
        users_tweets[uid].append( twt )

    for uid in users_tweets:
        if( len(users_tweets[uid]) > max_tweets ):
            # users_tweets[uid] = sorted( users_tweets[uid], key=lambda k: k['tweet_time'] )
            users_tweets[uid] = users_tweets[uid][:max_tweets]
            
    for uid in users_tweets:
        if( len(users_tweets[uid]) < min_tweets ):
            continue

        final_users_tweets[uid] = users_tweets[uid]

    return final_users_tweets

def get_info_ops_drivers_control_users_tweets(file_path, dataset_name, min_tweets_per_user, max_tweets_per_user, gen_bloc_params, all_bloc_symbols):
    control_path = os.path.join(file_path, dataset_name, 'DriversControl', 'control_driver_users.csv')
    driver_path = os.path.join(file_path, dataset_name, 'driver_tweets.csv.gz')

    # Load user posts
    driver_posts = get_driver_per_day_tweets(driver_path)
    control_posts = get_driver_control_per_day_tweets(control_path)

    campaign_start_date = min(convert_to_ISO_time(post['created_at']) for post in driver_posts['driver_posts'])
    campaign_last_date = max(convert_to_ISO_time(post['created_at']) for post in driver_posts['driver_posts'])

    stoping_criteria = True
    current_date = campaign_start_date
    end_of_two_weeks = campaign_start_date

    overall_driver_tweets = []
    overall_control_tweets = []
    
    selected_weeks = []
    driver_accounts = []
    while (stoping_criteria):
        start_of_two_weeks = current_date
        end_of_two_weeks = start_of_two_weeks + timedelta(days=14)

        current_date += timedelta(days=14)

        # Filter tweets for the 2-week window
        driver_week_tweets = [
            tweet for tweet in driver_posts['driver_posts']
            if 'created_at' in tweet
            and start_of_two_weeks <= convert_to_ISO_time(tweet['created_at']) <= end_of_two_weeks
        ]

        if len(driver_week_tweets) > 0:  
            selected_weeks.append((start_of_two_weeks, end_of_two_weeks))
            control_week_tweets = [
                tweet for tweet in control_posts['control_driver_posts']
                if 'created_at' in tweet
                and start_of_two_weeks <= convert_to_ISO_time(tweet['created_at']) <= end_of_two_weeks
            ]
            overall_control_tweets.extend(control_week_tweets)
            overall_driver_tweets.extend(driver_week_tweets)


        driver_users = get_all_per_user_tweets(overall_driver_tweets, max_tweets_per_user, min_tweets_per_user)
        user_accounts = list(driver_users.keys())
        new_ids = [uid for uid in user_accounts if uid not in driver_accounts]
        if new_ids:
            driver_accounts.extend(new_ids)
        
        if(campaign_last_date <= current_date):
            break
        if(len(driver_users)<10):
            continue
        else:
            scan_start = current_date
            tz = getattr(current_date, "tzinfo", None)
            year_end_exclusive = datetime(current_date.year + 1, 1, 1, tzinfo=tz)


            while scan_start < year_end_exclusive and scan_start <= campaign_last_date: 
                scan_end = min(scan_start + timedelta(days=14), year_end_exclusive)

                driver_week_tweets = [
                    tweet for tweet in driver_posts["driver_posts"]
                    if "created_at" in tweet
                    and scan_start <= convert_to_ISO_time(tweet["created_at"]) < scan_end
                ]

                if len(driver_week_tweets) > 0:
                    selected_weeks.append((scan_start, scan_end))
                    control_week_tweets = [
                        tweet for tweet in control_posts["control_driver_posts"]
                        if "created_at" in tweet
                        and scan_start <= convert_to_ISO_time(tweet["created_at"]) < scan_end
                    ]
                    overall_driver_tweets.extend(driver_week_tweets)
                    overall_control_tweets.extend(control_week_tweets)

                    driver_users = get_all_per_user_tweets(overall_driver_tweets, max_tweets_per_user, min_tweets_per_user)
                    user_accounts = list(driver_users.keys())
                    new_ids = [uid for uid in user_accounts if uid not in driver_accounts]
                    if new_ids:
                        driver_accounts.extend(new_ids)

                
                # advance to next 2-week window
                scan_start = scan_end
            end_of_two_weeks = scan_end
            break

        

    control_users = get_all_per_user_tweets(overall_control_tweets, max_tweets_per_user, min_tweets_per_user)
    driver_users = get_all_per_user_tweets(overall_driver_tweets, max_tweets_per_user, min_tweets_per_user)
    
    k = min(len(driver_accounts), len(control_users))
    keys_sorted_num = sorted(control_users, key=lambda k: int(k))
    random.seed(42)
    sample_keys = random.sample(keys_sorted_num, k)
    sampled_control_accounts = (
        {k: control_users[k] for k in sample_keys}
        if isinstance(control_users, dict) else sample_keys
    )

    job_lst = []
    for uid, tweets in sampled_control_accounts.items():
        job_lst.append({
            'func': add_bloc_sequences,
            'args': {'tweets': tweets, 'all_bloc_symbols': all_bloc_symbols, **gen_bloc_params},
            'print': '',
            'misc': None
        })
    all_control_blocs = parallelTask(job_lst, threadCount=5)

    records = []
    for ou in all_control_blocs:
        records.append({
            'user_id': ou['output']['user_id'],
            'user_class': 'control',
            'src': 'infoOps',
            'u_bloc': ou['output']
        })

    for uid, tweets in driver_users.items():
        job_lst.append({
            'func': add_bloc_sequences,
            'args': {'tweets': tweets, 'all_bloc_symbols': all_bloc_symbols, **gen_bloc_params},
            'print': '',
            'misc': None
        })
    all_driver_blocs = parallelTask(job_lst, threadCount=5)

    for ou in all_driver_blocs:
        records.append({
            'user_id': oou['output']['user_id'],
            'user_class': 'driver',
            'src': 'infoOps',
            'u_bloc': ou['output']
        })

    return records

def main(cfg):
    path_to_dataset = cfg['infoOps_dataset']
    src = cfg['src']
    
    for datasetName in src:
        print(f"Processing {name_mapper[datasetName]}...")
        file_location = os.path.join(path_to_dataset, datasetName)
        control = os.path.exists(os.path.join(file_location, 'DriversControl/control_driver_users.csv'))
        drivers = os.path.exists(os.path.join(file_location, 'driver_tweets.csv.gz'))
        
        if control and drivers:
            segmentation_type = cfg.get("segmentation_type")
            comparison_method = cfg.get("comparison_method")
            distance_metric = cfg.get("distance_metric")
            n_gram = cfg.get("n_gram")
            gen_bloc_params = cfg.get("gen_bloc_params", {})
            min_tweets_per_user = cfg.get("min_tweets_per_user", 20)
            max_tweets_per_user = cfg.get("max_tweets_per_user", 5000)
            all_bloc_symbols = get_default_symbols()

            records = get_info_ops_drivers_control_users_tweets(path_to_dataset, datasetName, min_tweets_per_user, max_tweets_per_user, gen_bloc_params, all_bloc_symbols)
            records = segment_bloc_for_all(records, segmentation_type, n_gram)
            records = calculate_changes_for_all(records, comparison_method, distance_metric)

            # build final user_data for classifier
            user_data = []
            for r in records:
                seg = r["segmented_bloc_string"]
                if(len(r["action_changes_list"]) >= 2 or len(r["content_changes_list"])  >= 2):
                    user_data.append(
                        {
                            "user_class": r["user_class"],
                            "src": r["src"],
                            "action_changes_list": r["action_changes_list"],
                            "content_changes_list": r["content_changes_list"],
                            "action_bloc": seg["action"],
                            "content_bloc": seg["content_syntactic"],
                            "user_id": r["user_id"],
                        }
                    )

            df = pd.DataFrame(user_data)
            best_score, min_class_size = classifier(df, "coordination_detection")
            print("Best F1:", best_score, "| min_class_size:", min_class_size)
            return best_score, min_class_size
        else:
            print(f"Dataset {name_mapper[datasetName]} is missing required files. Skipping...")
            continue