import csv
import os
import re
import glob
import copy
import numpy as np
# datastructures
from collections import defaultdict
# time objects
import calendar
import datetime
# progress bar
from tqdm.autonotebook import tqdm


# Define CONSTANTS
DATA_PATH = '../data/raw_data/'
TEMP_PATH = '../data/interim/'
FEATURE_PATH = os.path.join(DATA_PATH, 'US_Firms_Specific/')


def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])

    return datetime.date(year, month, day)


def instance_list(path):
    """ Parse instance keys:
    Args:
        path: (string) path to US_Firms_Specific file
    Returns:
        key_list, com_dict: (list, dict)
            (list) list of instance' tuples (Company Mapping No., datetime(Year, Month))
            (dict) dictionary of {Company Mapping No.: {start: datetime, end: datetime}}
    """
    key_list = []
    invalid_pd_set = set()
    com_dict = defaultdict(lambda: defaultdict())
    fea_dict = defaultdict(list)
    cpd_dict = defaultdict(list)
    files = glob.glob(os.path.join(path, 'US_Firms_Specific_*.csv'))
    pbar = tqdm(files)

    for _file in pbar:
        pbar.set_description("Processing %s" % os.path.basename(_file))
        num_line = int(os.popen('wc -l < ' + _file).read()[:-1])
        with open(_file, 'r') as f:
            reader = csv.DictReader(f)
            # FIXME hard coded feature column position
            f_cols = reader.fieldnames[3:17]
            pd_cols = reader.fieldnames[17:]
            for row in tqdm(reader, total= num_line - 1, desc='Parsing csv'):
                mapping = int(row['Company Mapping No.'])
                time_stamp = datetime.datetime.strptime(row['Year'] + row['Month'], '%Y%m').date()
                key = (mapping, time_stamp)
                fea = np.asarray([float(row[col]) for col in f_cols])
                cpd = np.asarray([float(row[key]) for key in pd_cols])
                if np.isnan(cpd).any():
                    invalid_pd_set.add((mapping, time_stamp))
                fea_dict[key] = fea
                cpd_dict[key] = cpd
                key_list.append(key)
                # record start/end time for each mapping
                if mapping not in com_dict.keys():
                    com_dict[mapping]['start'] = time_stamp
                    com_dict[mapping]['end'] = time_stamp
                else:
                    if time_stamp < com_dict[mapping]['start']:
                        com_dict[mapping]['start'] = time_stamp
                    elif time_stamp > com_dict[mapping]['end']:
                        com_dict[mapping]['end'] = time_stamp

    return key_list, fea_dict, cpd_dict, com_dict, invalid_pd_set


def event_parser(path):
    """
    """
    com_dict = defaultdict(lambda: defaultdict())
    file = os.path.join(path, 'File_Location_by_Firms.csv')
    num_line = int(os.popen('wc -l < ' + file).read()[:-1])
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total= num_line - 1, desc='Parsing csv'):
            mapping = int(row['Company Mapping No.'])
            # collapse event date into the beginning of month
            time_stamp = datetime.datetime.strptime(row['Exit Date'], '%Y%m%d').date().replace(day=1)
            if mapping not in com_dict.keys():
                com_dict[mapping]['date'] = time_stamp
                com_dict[mapping]['event'] = int(row['Event Type'])
            else:
                raise ValueError('Duplicated event for Mapping ID:{}, plz check'.format(mapping))

    return com_dict


def fix_enddate(key_list, com_dict, event_dict):
    """
    Args:
        key_list: (list) list of tuple
        com_dict: (dict)
        event_dict: (dict)
    Returns:
        new_dict: (dict) modified com_dict, where the end dates changed to (event_date - 1)

    """
    #assert len(event_dict.keys()) == len(com_dict.keys()), '# of Mapping ID is not compatible'
    drop_cnt = 0
    new_dict = copy.deepcopy(com_dict)  # dict is mutable object
    pbar = tqdm(key_list)
    for key, time in pbar:
        pbar.set_description("checking %s" % key)
        eve = event_dict[key]['date']
        if time < eve:  # Rule: event must occur after collected data 
            pass
        else:
            delay = add_months(eve, -1)
            if new_dict[key]['end'] == delay:
                pass
            else:
                new_dict[key]['end'] = delay
            drop_cnt += 1
    print('{} instances is invalid'.format(drop_cnt))

    return new_dict


def build_event(key_list, com_dict, event_dict, cumulative=True, f_month=[]):
    print('building label list')
    label_dict = defaultdict(list)
    if f_month == []:
        print('Use default forward list: range(6)')
        f_month = [ i + 1 for i in range(6)]
    for mapping, time in tqdm(key_list):
        event_list = []
        event_date = event_dict[mapping]['date']
        event_type = event_dict[mapping]['event']
        end_date = add_months(com_dict[mapping]['end'], 1)
        for month in f_month:
            position = add_months(time, month)
            if cumulative:
                condition = position >= event_date
            else:
                condition = position == event_date

            if time >= end_date:
                event_list.append(-1)  # for abnormal data, event occured but still have record
            else:
                if condition and (event_type != 0):
                    event_list.append(event_type)
                else:
                    if position <= end_date:
                        event_list.append(0)
                    else:
                        event_list.append(-1)
        label_dict[(mapping, time)] = event_list

    return label_dict


def write_dict(data, name=''):
    """TODO: Docstring for write_csv.
    :returns: TODO

    """
    result = os.path.join(TEMP_PATH, name)
    with open(result, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in data.items():
            mappin_id, datetime = key
            year = datetime.year
            month = datetime.month
            if type(value) == np.ndarray:
                value = value.tolist()
            writer.writerow([mappin_id, year, month] + value)


def main():
    """ Main Function
    """
    key_list, fea_dict, cpd_dict, com_dict, invalid_pd_set = instance_list(FEATURE_PATH)
    for invalid in invalid_pd_set:
        fea_dict.pop(invalid)
        cpd_dict.pop(invalid)
    key_list = [key for key in key_list if key not in invalid_pd_set]
    print("{} instances masked by CPD is NaN".format(len(invalid_pd_set)))
    event_dict = event_parser(DATA_PATH)
    fixed_com_dict = fix_enddate(key_list, com_dict, event_dict)
    cum_dict = build_event(key_list, fixed_com_dict, event_dict,
            cumulative=True,
            f_month=[1,3,6,12,24,36,48,60])
    for_dict = build_event(key_list, fixed_com_dict, event_dict,
            cumulative=False,
            f_month=[i + 1 for i in range(60)])
    # File I/O
    write_dict(cum_dict, name = 'label_cum.csv')
    write_dict(for_dict, name = 'label_for.csv')
    write_dict(fea_dict, name = 'data_fea.csv')
    write_dict(cpd_dict, name = 'data_cpd.csv')




if __name__ == "__main__":
    main()

