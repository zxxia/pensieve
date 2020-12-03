import os

import numpy as np
from constants import (B_IN_MB, BITRATE_LEVELS, BITS_IN_BYTE, BUFFER_THRESH,
                       DRAIN_BUFFER_SLEEP_TIME, LINK_RTT,
                       MILLISECONDS_IN_SECOND, NOISE_HIGH, NOISE_LOW,
                       PACKET_PAYLOAD_PORTION, RANDOM_SEED, TOTAL_VIDEO_CHUNK,
                       VIDEO_BIT_RATE, VIDEO_CHUNK_LEN)

VIDEO_SIZE_FILE = '../data/video_sizes/video_size_'


class BaseEnvironment:
    def __init__(self, link_rtt, buffer_thresh, drain_buffer_sleep_time,
                 packet_payload_portion):
        self.link_rtt = link_rtt
        self.buffer_thresh = buffer_thresh
        self.drain_buffer_sleep_time = drain_buffer_sleep_time
        self.packet_payload_portion = packet_payload_portion

    def get_video_chunk(self, quality):
        raise NotImplementedError


class NetworkEnvironment(BaseEnvironment):
    def __init__(self, trace_time, trace_bw, trace_file_name,
                 video_size_file_dir, trace_video_same_duration_flag=False,
                 link_rtt=LINK_RTT, buffer_thresh=BUFFER_THRESH,
                 drain_buffer_sleep_time=DRAIN_BUFFER_SLEEP_TIME,
                 packet_payload_portion=PACKET_PAYLOAD_PORTION, fixed=True):
        super().__init__(link_rtt, buffer_thresh, drain_buffer_sleep_time,
                         packet_payload_portion)
        self.trace_time = trace_time
        self.trace_bw = trace_bw
        self.trace_file_name = trace_file_name
        self.trace_video_same_duration_flag = trace_video_same_duration_flag
        self.trace_ptr = 1
        self.last_trace_ts = self.trace_time[self.trace_ptr - 1]
        self.nb_chunk_sent = 0
        self.buffer_size = 0
        self.fixed = fixed

        self.video_size = {}  # in bytes
        for bitrate in range(len(VIDEO_BIT_RATE)):
            self.video_size[bitrate] = []
            video_size_file = os.path.join(video_size_file_dir,
                                           'video_size_{}'.format(bitrate))
            with open(video_size_file, 'r') as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

        if trace_video_same_duration_flag:
            self.total_video_chunk = max(
                TOTAL_VIDEO_CHUNK,
                self.trace_time[-1]*MILLISECONDS_IN_SECOND//VIDEO_CHUNK_LEN)
        else:
            self.total_video_chunk = TOTAL_VIDEO_CHUNK

    def get_video_chunk(self, quality):
        assert 0 <= quality < len(VIDEO_BIT_RATE)

        video_chunk_size = self.video_size[quality][
            self.nb_chunk_sent % TOTAL_VIDEO_CHUNK]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        bytes_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            # throughput = bytes per ms
            throughput = self.trace_bw[self.trace_ptr] * B_IN_MB / BITS_IN_BYTE
            duration = self.trace_time[self.trace_ptr] - self.last_trace_ts

            packet_payload = throughput * duration * self.packet_payload_portion

            if bytes_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - bytes_sent) / \
                    throughput / self.packet_payload_portion
                delay += fractional_time
                self.last_trace_ts += fractional_time
                assert(self.last_trace_ts <= self.trace_time[self.trace_ptr])
                break

            bytes_sent += packet_payload
            delay += duration
            self.last_trace_ts = self.trace_time[self.trace_ptr]
            self.trace_ptr += 1

            if self.trace_ptr >= len(self.trace_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.trace_ptr = 1
                self.last_trace_ts = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += self.link_rtt

        # add a multiplicative noise to the delay
        if not self.fixed:
            delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > self.buffer_thresh:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - self.buffer_thresh
            sleep_time = np.ceil(drain_buffer_time /
                                 self.drain_buffer_sleep_time) * \
                self.drain_buffer_sleep_time
            self.buffer_size -= sleep_time

            while True:
                duration = self.trace_time[self.trace_ptr] - self.last_trace_ts
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_trace_ts += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_trace_ts = self.trace_time[self.trace_ptr]
                self.trace_ptr += 1

                if self.trace_ptr >= len(self.trace_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.trace_ptr = 1
                    self.last_trace_ts = self.trace_time[self.trace_ptr - 1]

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.nb_chunk_sent += 1
        video_chunk_remain = self.total_video_chunk - self.nb_chunk_sent

        end_of_video = self.nb_chunk_sent >= self.total_video_chunk

        next_video_chunk_sizes = []
        for i in range(len(VIDEO_BIT_RATE)):
            next_video_chunk_sizes.append(
                self.video_size[i][self.nb_chunk_sent % TOTAL_VIDEO_CHUNK])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, all_file_names=None,
                 random_seed=RANDOM_SEED, fixed=False):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)
        self.fixed = fixed
        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw
        self.all_file_names = all_file_names

        self.video_chunk_counter = 0
        self.buffer_size = 0

        # pick a random trace file
        self.trace_idx = 0 if fixed else np.random.randint(
            len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        self.mahimahi_start_ptr = 1
        self.mahimahi_ptr = 1 if fixed else np.random.randint(
            1, len(self.cooked_bw))
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))
        self.total_video_chunk = TOTAL_VIDEO_CHUNK

    def get_video_chunk(self, quality):

        assert 0 <= quality < BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            # throughput = bytes per ms
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                    throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert(self.last_mahimahi_time <=
                       self.cooked_time[self.mahimahi_ptr])
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        # add a multiplicative noise to the delay
        if not self.fixed:
            delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                    - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = self.total_video_chunk - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.total_video_chunk:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            if self.fixed:
                self.trace_idx += 1
                if self.trace_idx >= len(self.all_cooked_time):
                    self.trace_idx = 0
            else:
                # pick a random trace file
                self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = self.mahimahi_start_ptr if self.fixed else np.random.randint(
                1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(
                self.video_size[i][self.video_chunk_counter])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain


class EnvironmentNoRandomStart:
    def __init__(self, all_cooked_time, all_cooked_bw, all_file_names=None,
                 random_seed=RANDOM_SEED, fixed=False):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)
        self.fixed = fixed
        self.all_trace_time = all_cooked_time
        self.all_trace_bw = all_cooked_bw
        self.all_file_names = all_file_names

        self.video_chunk_counter = 0
        self.buffer_size = 0

        # pick a random trace file
        self.trace_idx = 0 if fixed else np.random.randint(
            len(self.all_trace_time))  # index among all traces
        # trace ptr points a position on the network trace
        self.trace_ptr = 1 if fixed else np.random.randint(
            1, len(self.trace_bw))
        self.trace_time = self.all_trace_time[self.trace_idx]
        self.trace_bw = self.all_trace_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.last_trace_ts = self.trace_time[self.trace_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))
        self.total_video_chunk = max(
            TOTAL_VIDEO_CHUNK,
            self.trace_time[-1]*MILLISECONDS_IN_SECOND//VIDEO_CHUNK_LEN)
        self.buffer_thresh = np.random.randint(4, 20) * 1000

    def get_video_chunk(self, quality):

        assert 0 <= quality < BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][
            self.video_chunk_counter % TOTAL_VIDEO_CHUNK]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        bytes_sent = 0  # bytes sent in current video chunk

        while True:  # download video chunk over mahimahi
            # throughput = bytes per ms
            throughput = self.trace_bw[self.trace_ptr] * B_IN_MB / BITS_IN_BYTE
            duration = self.trace_time[self.trace_ptr] - self.last_trace_ts

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if bytes_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - bytes_sent) / \
                    throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_trace_ts += fractional_time
                assert(self.last_trace_ts <= self.trace_time[self.trace_ptr])
                break

            bytes_sent += packet_payload
            delay += duration
            self.last_trace_ts = self.trace_time[self.trace_ptr]
            self.trace_ptr += 1

            if self.trace_ptr >= len(self.trace_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.trace_ptr = 1
                self.last_trace_ts = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        # add a multiplicative noise to the delay
        if not self.fixed:
            delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNK_LEN

        # sleep if buffer gets too large
        # buffer is too full and we wait for some data to be consumed
        sleep_time = 0
        if self.buffer_size > self.buffer_thresh:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - self.buffer_thresh
            sleep_time = np.ceil(drain_buffer_time/DRAIN_BUFFER_SLEEP_TIME) * \
                DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.trace_time[self.trace_ptr] - self.last_trace_ts
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_trace_ts += (sleep_time / MILLISECONDS_IN_SECOND)
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_trace_ts = self.trace_time[self.trace_ptr]
                self.trace_ptr += 1

                if self.trace_ptr >= len(self.trace_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.trace_ptr = 1
                    self.last_trace_ts = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = self.total_video_chunk - self.video_chunk_counter

        end_of_video = self.video_chunk_counter >= self.total_video_chunk
        if end_of_video:
            self.buffer_size = 0
            self.video_chunk_counter = 0

            if self.fixed:
                self.trace_idx = (self.trace_idx+1) % len(self.all_trace_time)
            else:
                # pick a random trace file
                self.trace_idx = np.random.randint(len(self.all_trace_time))
            self.trace_time = self.all_trace_time[self.trace_idx]
            self.trace_bw = self.all_trace_bw[self.trace_idx]
            self.total_video_chunk = max(TOTAL_VIDEO_CHUNK,
                                         self.trace_time[-1]//4)

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.trace_ptr = 1
            self.last_trace_ts = self.trace_time[self.trace_ptr - 1]
            self.buffer_thresh = np.random.randint(4, 20) * 1000
            while self.buffer_thresh == 4000 or self.buffer_thresh == 8000 or \
                    self.buffer_thresh == 16000 or \
                    self.buffer_thresh == 24000 or \
                    self.buffer_thresh == 32000 or \
                    self.buffer_thresh == 60000 or \
                    self.buffer_thresh == 120000:
                self.buffer_thresh = np.random.randint(4, 20) * 1000
        # print(self.buffer_size, self.buffer_thresh)
        # assert 40*1000 <= self.buffer_thresh <= 60 * 1000

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(
                self.video_size[i][
                    self.video_chunk_counter % TOTAL_VIDEO_CHUNK])

        return delay, sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, video_chunk_size, \
            next_video_chunk_sizes, end_of_video, video_chunk_remain


# class EnvironmentConstantDelay:
#     # TODO: implement constant delay
#     def __init__(self, all_cooked_time, all_cooked_bw, all_file_names=None,
#                  random_seed=RANDOM_SEED, fixed=False):
#         assert len(all_cooked_time) == len(all_cooked_bw)
#
#         np.random.seed(random_seed)
#         self.fixed = fixed
#         self.all_trace_time = all_cooked_time
#         self.all_trace_bw = all_cooked_bw
#         self.all_file_names = all_file_names
#
#         self.video_chunk_counter = 0
#         self.buffer_size = 0
#
#         # pick a random trace file
#         self.trace_idx = 0 if fixed else np.random.randint(
#             len(self.all_trace_time))  # index among all traces
#         self.trace_time = self.all_trace_time[self.trace_idx]
#         self.trace_bw = self.all_trace_bw[self.trace_idx]
#
#         # mahimahi ptr points a position on the network trace
#         self.mahimahi_ptr = 1 if fixed else np.random.randint(
#             1, len(self.trace_bw))
#         # randomize the start point of the trace
#         # note: trace file starts with time 0
#         self.last_mahimahi_time = self.trace_time[self.mahimahi_ptr - 1]
#
#         self.video_size = {}  # in bytes
#         for bitrate in range(BITRATE_LEVELS):
#             self.video_size[bitrate] = []
#             with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
#                 for line in f:
#                     self.video_size[bitrate].append(int(line.split()[0]))
#         self.total_video_chunk = max(TOTAL_VIDEO_CHUNK, self.trace_time[-1]//4)
#
#     def get_video_chunk(self, quality):
#
#         assert 0 <= quality < BITRATE_LEVELS
#
#         video_chunk_size = self.video_size[quality][
#             self.video_chunk_counter % TOTAL_VIDEO_CHUNK]
#
#         # use the delivery opportunity in mahimahi
#         delay = 0.0  # in ms
#         bytes_sent = 0  # bytes sent in current video chunk
#
#         while True:  # download video chunk over mahimahi
#             # throughput = bytes per ms
#             throughput = (self.trace_bw[self.mahimahi_ptr] * B_IN_MB /
#                           BITS_IN_BYTE)
#             # duration = self.trace_time[self.mahimahi_ptr] \
#             #     - self.last_mahimahi_time
#             duration = 0
#
#             packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION
#
#             if bytes_sent + packet_payload > video_chunk_size:
#
#                 fractional_time = (video_chunk_size - bytes_sent) / \
#                     throughput / PACKET_PAYLOAD_PORTION
#                 delay += fractional_time
#                 self.last_mahimahi_time += fractional_time
#                 # assert(self.last_mahimahi_time <=
#                 #        self.trace_time[self.mahimahi_ptr]), "last_mahimahi_time={}, fractional_time={}, current_time={}".format(
#                 #         self.last_mahimahi_time, fractional_time,
#                 #         self.trace_time[self.mahimahi_ptr])
#                 break
#
#             bytes_sent += packet_payload
#             delay += duration
#             self.last_mahimahi_time = self.trace_time[self.mahimahi_ptr]
#             self.mahimahi_ptr += 1
#
#             if self.mahimahi_ptr >= len(self.trace_bw):
#                 # loop back in the beginning
#                 # note: trace file starts with time 0
#                 self.mahimahi_ptr = 1
#                 self.last_mahimahi_time = 0
#
#         delay *= MILLISECONDS_IN_SECOND
#         delay += LINK_RTT
#
#         # add a multiplicative noise to the delay
#         if not self.fixed:
#             delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)
#
#         # rebuffer time
#         rebuf = np.maximum(delay - self.buffer_size, 0.0)
#
#         # update the buffer
#         self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)
#
#         # add in the new chunk
#         self.buffer_size += VIDEO_CHUNK_LEN
#
#         # sleep if buffer gets too large
#         # buffer is too full and we wait for some data to be consumed
#         sleep_time = 0
#         if self.buffer_size > BUFFER_THRESH:
#             # exceed the buffer limit
#             # we need to skip some network bandwidth here
#             # but do not add up the delay
#             drain_buffer_time = self.buffer_size - BUFFER_THRESH
#             sleep_time = np.ceil(drain_buffer_time/DRAIN_BUFFER_SLEEP_TIME) * \
#                 DRAIN_BUFFER_SLEEP_TIME
#             self.buffer_size -= sleep_time
#
#             while True:
#                 duration = self.trace_time[self.mahimahi_ptr] - \
#                     self.last_mahimahi_time
#                 if duration > sleep_time / MILLISECONDS_IN_SECOND:
#                     self.last_mahimahi_time += (sleep_time /
#                                                 MILLISECONDS_IN_SECOND)
#                     break
#                 sleep_time -= duration * MILLISECONDS_IN_SECOND
#                 self.last_mahimahi_time = self.trace_time[self.mahimahi_ptr]
#                 self.mahimahi_ptr += 1
#
#                 if self.mahimahi_ptr >= len(self.trace_bw):
#                     # loop back in the beginning
#                     # note: trace file starts with time 0
#                     self.mahimahi_ptr = 1
#                     self.last_mahimahi_time = 0
#
#         # the "last buffer size" return to the controller
#         # Note: in old version of dash the lowest buffer is 0.
#         # In the new version the buffer always have at least
#         # one chunk of video
#         return_buffer_size = self.buffer_size
#
#         self.video_chunk_counter += 1
#         video_chunk_remain = self.total_video_chunk - self.video_chunk_counter
#
#         end_of_video = False
#         if video_chunk_remain == 0:
#             end_of_video = True
#             self.buffer_size = 0
#             self.video_chunk_counter = 0
#
#             if self.fixed:
#                 self.trace_idx = (self.trace_idx+1) % len(self.all_trace_time)
#             else:
#                 # pick a random trace file
#                 self.trace_idx = np.random.randint(len(self.all_trace_time))
#             self.trace_time = self.all_trace_time[self.trace_idx]
#             self.trace_bw = self.all_trace_bw[self.trace_idx]
#             self.total_video_chunk = max(TOTAL_VIDEO_CHUNK,
#                                          self.trace_time[-1]//4)
#
#             # randomize the start point of the video
#             # note: trace file starts with time 0
#             self.mahimahi_ptr = 1
#             self.last_mahimahi_time = self.trace_time[self.mahimahi_ptr - 1]
#
#         next_video_chunk_sizes = []
#         for i in range(BITRATE_LEVELS):
#             next_video_chunk_sizes.append(
#                 self.video_size[i][
#                     self.video_chunk_counter % TOTAL_VIDEO_CHUNK])
#
#         return delay, sleep_time, \
#             return_buffer_size / MILLISECONDS_IN_SECOND, \
#             rebuf / MILLISECONDS_IN_SECOND, video_chunk_size, \
#             next_video_chunk_sizes, end_of_video, video_chunk_remain
#
#
# class EnvironmentNoSleep:
#     def __init__(self, all_cooked_time, all_cooked_bw, all_file_names=None,
#                  random_seed=RANDOM_SEED, fixed=False):
#         assert len(all_cooked_time) == len(all_cooked_bw)
#
#         np.random.seed(random_seed)
#         self.fixed = fixed
#         self.all_trace_time = all_cooked_time
#         self.all_trace_bw = all_cooked_bw
#         self.all_file_names = all_file_names
#
#         self.video_chunk_counter = 0
#         self.buffer_size = 0
#
#         # pick a random trace file
#         self.trace_idx = 0 if fixed else np.random.randint(
#             len(self.all_trace_time))  # index among all traces
#         self.trace_time = self.all_trace_time[self.trace_idx]
#         self.trace_bw = self.all_trace_bw[self.trace_idx]
#
#         # mahimahi ptr points a position on the network trace
#         self.mahimahi_ptr = 1 if fixed else np.random.randint(
#             1, len(self.trace_bw))
#         # randomize the start point of the trace
#         # note: trace file starts with time 0
#         self.last_mahimahi_time = self.trace_time[self.mahimahi_ptr - 1]
#
#         self.video_size = {}  # in bytes
#         for bitrate in range(BITRATE_LEVELS):
#             self.video_size[bitrate] = []
#             with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
#                 for line in f:
#                     self.video_size[bitrate].append(int(line.split()[0]))
#         self.total_video_chunk = max(TOTAL_VIDEO_CHUNK, self.trace_time[-1]//4)
#
#     def get_video_chunk(self, quality):
#
#         assert 0 <= quality < BITRATE_LEVELS
#
#         video_chunk_size = self.video_size[quality][
#             self.video_chunk_counter % TOTAL_VIDEO_CHUNK]
#
#         # use the delivery opportunity in mahimahi
#         delay = 0.0  # in ms
#         bytes_sent = 0  # bytes sent in current video chunk
#
#         while True:  # download video chunk over mahimahi
#             # throughput = bytes per ms
#             throughput = (self.trace_bw[self.mahimahi_ptr] * B_IN_MB /
#                           BITS_IN_BYTE)
#             duration = self.trace_time[self.mahimahi_ptr] \
#                 - self.last_mahimahi_time
#
#             packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION
#
#             if bytes_sent + packet_payload > video_chunk_size:
#
#                 fractional_time = (video_chunk_size - bytes_sent) / \
#                     throughput / PACKET_PAYLOAD_PORTION
#                 delay += fractional_time
#                 self.last_mahimahi_time += fractional_time
#                 assert(self.last_mahimahi_time <=
#                        self.trace_time[self.mahimahi_ptr])
#                 break
#
#             bytes_sent += packet_payload
#             delay += duration
#             self.last_mahimahi_time = self.trace_time[self.mahimahi_ptr]
#             self.mahimahi_ptr += 1
#
#             if self.mahimahi_ptr >= len(self.trace_bw):
#                 # loop back in the beginning
#                 # note: trace file starts with time 0
#                 self.mahimahi_ptr = 1
#                 self.last_mahimahi_time = 0
#
#         delay *= MILLISECONDS_IN_SECOND
#         delay += LINK_RTT
#
#         # add a multiplicative noise to the delay
#         if not self.fixed:
#             delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)
#
#         # rebuffer time
#         rebuf = np.maximum(delay - self.buffer_size, 0.0)
#
#         # update the buffer
#         self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)
#
#         # add in the new chunk
#         self.buffer_size += VIDEO_CHUNK_LEN
#
#         # sleep if buffer gets too large
#         # buffer is too full and we wait for some data to be consumed
#         sleep_time = 0
#         # if self.buffer_size > BUFFER_THRESH:
#         #     # exceed the buffer limit
#         #     # we need to skip some network bandwidth here
#         #     # but do not add up the delay
#         #     drain_buffer_time = self.buffer_size - BUFFER_THRESH
#         #     sleep_time = np.ceil(drain_buffer_time/DRAIN_BUFFER_SLEEP_TIME) * \
#         #         DRAIN_BUFFER_SLEEP_TIME
#         #     self.buffer_size -= sleep_time
#         #
#         #     while True:
#         #         duration = self.trace_time[self.mahimahi_ptr] - \
#         #             self.last_mahimahi_time
#         #         if duration > sleep_time / MILLISECONDS_IN_SECOND:
#         #             self.last_mahimahi_time += (sleep_time /
#         #                                         MILLISECONDS_IN_SECOND)
#         #             break
#         #         sleep_time -= duration * MILLISECONDS_IN_SECOND
#         #         self.last_mahimahi_time = self.trace_time[self.mahimahi_ptr]
#         #         self.mahimahi_ptr += 1
#         #
#         #         if self.mahimahi_ptr >= len(self.trace_bw):
#         #             # loop back in the beginning
#         #             # note: trace file starts with time 0
#         #             self.mahimahi_ptr = 1
#         #             self.last_mahimahi_time = 0
#
#         # the "last buffer size" return to the controller
#         # Note: in old version of dash the lowest buffer is 0.
#         # In the new version the buffer always have at least
#         # one chunk of video
#         return_buffer_size = self.buffer_size
#
#         self.video_chunk_counter += 1
#         video_chunk_remain = self.total_video_chunk - self.video_chunk_counter
#
#         end_of_video = False
#         if video_chunk_remain == 0:
#             end_of_video = True
#             self.buffer_size = 0
#             self.video_chunk_counter = 0
#
#             if self.fixed:
#                 self.trace_idx = (self.trace_idx+1) % len(self.all_trace_time)
#             else:
#                 # pick a random trace file
#                 self.trace_idx = np.random.randint(len(self.all_trace_time))
#             self.trace_time = self.all_trace_time[self.trace_idx]
#             self.trace_bw = self.all_trace_bw[self.trace_idx]
#             self.total_video_chunk = max(TOTAL_VIDEO_CHUNK,
#                                          self.trace_time[-1]//4)
#
#             # randomize the start point of the video
#             # note: trace file starts with time 0
#             self.mahimahi_ptr = 1
#             self.last_mahimahi_time = self.trace_time[self.mahimahi_ptr - 1]
#
#         next_video_chunk_sizes = []
#         for i in range(BITRATE_LEVELS):
#             next_video_chunk_sizes.append(
#                 self.video_size[i][
#                     self.video_chunk_counter % TOTAL_VIDEO_CHUNK])
#
#         return delay, sleep_time, \
#             return_buffer_size / MILLISECONDS_IN_SECOND, \
#             rebuf / MILLISECONDS_IN_SECOND, video_chunk_size, \
#             next_video_chunk_sizes, end_of_video, video_chunk_remain
