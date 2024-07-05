def read_large_pgn(file_path, num_lines=100):
    """Генератор для построчного чтения большого PGN файла."""
    with open(file_path, 'r', encoding='utf-8') as file:
        for _ in range(num_lines):
            line = file.readline()
            if not line:
                break
            yield line

# Пример использования
pgn_file_path = 'E:/sky/lichess_db_standard_rated_2018-06.pgn'

for line in read_large_pgn(pgn_file_path, num_lines=100):
    print(line.strip())
