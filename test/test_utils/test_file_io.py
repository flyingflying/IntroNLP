# Author: lqxu


def test_big_file_reader():
    import os
    from core.utils import BigFileReader, DATA_DIR

    file_path = os.path.join(DATA_DIR, "sentence_embeddings/STS-B/cnsd-sts-train.txt")

    reader = BigFileReader(file_path)

    print(reader.read_specified_line(0))
    print(reader.read_specified_line(len(reader) - 1))
    print(reader.read_specified_line(1))


if __name__ == '__main__':
    test_big_file_reader()
