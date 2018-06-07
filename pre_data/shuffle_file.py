# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2018/6/6
import tensorflow as tf


def create_shuffle_files():
    files = [(("file%d") % i) for i in range(10)]
    print("\n files == %s" % (files))

    filenames_once = tf.train.match_filenames_once("*.*")
    with tf.Session() as sess:
        initializer = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        sess.run(initializer)
        print(sess.run(filenames_once))


def file_queue():
    filename_queue = tf.train.string_input_producer(["accounts1.csv", "accounts2.csv", "accounts3.csv"])
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    print("key = %s val = %s" % (key, value))

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [[1], [1], [1], [1], [1]]
    col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
    features = tf.concat([col1, col2, col3, col4], 0)

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(1200):
            # Retrieve a single instance:
            example, label = sess.run([features, col5])

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # create_shuffle_files()
    file_queue()
