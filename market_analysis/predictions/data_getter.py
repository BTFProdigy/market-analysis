from cassandra.cluster import Cluster
import datetime
import math
from collections import namedtuple
from datetime import datetime as dt
import pandas as pd
Value = namedtuple('Values', ['timestamp', 'name', 'value'])
ValueForParameter = namedtuple('Values', ['timestamp', 'value'])
Device = namedtuple('Device', ['id', 'type'])


class DataGetter(object):
    def __init__(self):
        self.cluster = Cluster(['192.168.84.132'])
        # self.cluster = Cluster(['localhost'])
        self.session = self.cluster.connect('presto')

    def get_values_for_parameter(self, device_id, parameter_name):
        query = """SELECT bucket, ts_offset, parameter_value FROM datapoints 
                   WHERE device_id = %(device_id)s 
                   AND parameter_name = %(parameter_name)s 
                   ALLOW FILTERING;"""

        params = {
            'device_id': device_id,
            "parameter_name": parameter_name
        }
        rows = self.session.execute(query, params)
        values = [ValueForParameter (self.create_extended_timestamp(row.bucket, row.ts_offset), row.parameter_value) for row in rows]

        return values


    def get_all_values(self, device_id):
        query = '''SELECT bucket, ts_offset, parameter_name, parameter_value FROM data 
                   WHERE device_id = %(device_id)s 
                   ALLOW FILTERING;'''

        params = {
            'device_id': device_id
        }

        rows = self.session.execute(query, params)

        values = [{"timestamp" : self.create_extended_timestamp(row.bucket, row.ts_offset),
                   "name" : row.parameter_name,
                   "value" : row.parameter_value} for row in rows]
        return values
        # return sorted(datapoints, key=lambda x: x.timestamp)

    def get_all_values_for_period_by_days(self, device_id, from_day, to_day):

        query = '''SELECT bucket, ts_offset, parameter_name, parameter_value FROM data
                   WHERE bucket >= %(from_bucket)s AND bucket <= %(to_bucket)s
                   AND device_id = %(device_id)s
                   ALLOW FILTERING;'''

        params = {
            'device_id': device_id,
            'from_bucket': from_day + " 00:00:00",
            'to_bucket': to_day + " 23:00:00"
        }

        rows = self.session.execute(query, params)

        values = [{"timestamp" : self.create_extended_timestamp(row.bucket, row.ts_offset),
                   "name" : row.parameter_name,
                   "value" : row.parameter_value} for row in rows]
        return values

    # works for buckets
    def get_all_values_for_period_by_buckets(self, device_id, from_timestamp, to_timestamp = None):

        query = '''SELECT bucket, ts_offset, parameter_name, parameter_value FROM data 
                   WHERE device_id = %(device_id)s 
                   AND bucket >= %(from_timestamp)s''' + ('' if to_timestamp == None else 'AND bucket <= %(to_timestamp)s') + ' ALLOW FILTERING;'

        params = {
            'device_id': device_id,
            'from_timestamp': from_timestamp,
            'to_timestamp' : to_timestamp
        }

        rows = self.session.execute(query, params)

        values = [{"timestamp" : self.create_extended_timestamp(row.bucket, row.ts_offset),
                   "name" : row.parameter_name,
                   "value" : row.parameter_value} for row in rows]
        return values

    def get_all_values_for_period_by_buckets_for_parameter(self, device_id, from_timestamp, parameter, to_timestamp = None):

        query = '''SELECT bucket, ts_offset, parameter_name, parameter_value FROM data 
                   WHERE device_id = %(device_id)s 
                   AND parameter_name = %(parameter)s 
                   AND bucket >= %(from_timestamp)s''' + ('' if to_timestamp == None else 'AND bucket <= %(to_timestamp)s') + ' ALLOW FILTERING;'

        params = {
            'device_id': device_id,
            'parameter': parameter,
            'from_timestamp': from_timestamp,
            'to_timestamp' : to_timestamp
        }

        rows = self.session.execute(query, params)

        values = [{"timestamp" : self.create_extended_timestamp(row.bucket, row.ts_offset),
                   "name" : row.parameter_name,
                   "value" : row.parameter_value} for row in rows]
        return values

    def get_all_values_for_period(self, device_id, from_timestamp, to_timestamp, parameter):
        bucket_from = from_timestamp
        bucket_to = to_timestamp
        bucket_from = bucket_from[0:14] + "00:00"
        bucket_to = bucket_to[0:14] + "00:00"

        from_ts_offset = self.get_offset_in_ms(from_timestamp)
        to_ts_offset = self.get_offset_in_ms(to_timestamp)

        # to_bucket = to_timestamp
        # to_bucket[14:19] = "00:00"

        query = '''SELECT bucket, ts_offset, parameter_name, parameter_value FROM data 
                   WHERE device_id = %(device_id)s 
                   AND parameter_name = %(parameter)s 
                   AND bucket >= %(from_bucket)s AND ts_offset >= %(from_ts_offset)s 
                   AND bucket<= %(to_bucket)s and ts_offset <= %(to_ts_offset)s ALLOW FILTERING;'''

        params = {
            'device_id': device_id,
            'parameter': parameter,
            'from_bucket': bucket_from,
            'to_bucket': bucket_to,
            'from_ts_offset': from_ts_offset,
            'to_ts_offset' : to_ts_offset
        }

        rows = self.session.execute(query, params)

        values = [{"timestamp" : self.create_extended_timestamp(row.bucket, row.ts_offset),
                   "name" : row.parameter_name,
                   "value" : row.parameter_value} for row in rows]
        return values


    def insert_predictions(self, device_id, parameter, last_timestamp, offset_in_seconds, predictions, mode):
        if mode=="general":
            table = "predictions"
        else:
            table = "predictions_test"

        for index, prediction in enumerate(predictions):
            prediction = float(prediction)
            timestamp = self.get_future_timestamp(last_timestamp, offset_in_seconds, index)
            print "Future timestamp"
            print timestamp

            query = '''INSERT INTO ''' + table + ''' (device, timestamp, timestamp_created, parameter, value, prediction_error) 
                      VALUES (%(device_id)s, %(timestamp)s, %(timestamp_created)s, %(parameter)s, %(val)s, -1);'''
            params = {
                'device_id': device_id,
                'timestamp' : str(timestamp),
                'timestamp_created': pd.to_datetime(str(last_timestamp)).strftime("%Y-%m-%d %H:%M:%S"),
                'parameter': parameter,
                'val': prediction
            }

            self.session.execute(query, params)
        return


    def get_future_timestamp(self, last_timestamp, offset, index):
        future_timestamp = datetime.datetime.utcfromtimestamp(last_timestamp.astype('O')/1e9) + datetime.timedelta(seconds = int(offset)*(index +1))

        # if datetime formatted timestamp
        # future_timestamp = datetime.datetime.strptime(last_timestamp, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(seconds = int(offset)*(index +1))
        return future_timestamp


    def create_bucket_timestamp(self, timestamp):
        parsed_timestamp = dt.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        timestamp_bucket = parsed_timestamp.replace(minute = 0, second = 0)
        return timestamp_bucket.strftime("%Y-%m-%d %H:%M:%S")

    def create_extended_timestamp(self, bucket, offset_ms):
        offset_seconds = offset_ms / 1000
        minute = offset_seconds / 60
        second = offset_seconds - minute * 60
        minute_str = str(minute).zfill(2)
        second_str = str(second).zfill(2)
        return '{}:{}:{}'.format(bucket[:13], minute_str, second_str)

    def get_offset_in_ms(self, time):
        minutes = int(time[14:16])
        seconds = int(time[17:19])

        miliseconds = seconds*1000 + minutes*60*1000

        return miliseconds

    def get_prediction(self, device_id, timestamp):
        query = '''SELECT parameter, value FROM predictions
                   WHERE timestamp = %(timestamp)s  
                   AND device = %(device)s
                   ALLOW FILTERING;'''

        params = {
            'device': device_id,
            'timestamp': timestamp
        }

        rows = self.session.execute(query, params)

        values = [{
            "parameter" : row.parameter,
            "value" : row.value} for row in rows]
        return values
    # lag order is usually small
