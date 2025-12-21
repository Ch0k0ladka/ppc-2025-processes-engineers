# Линейка

**Студент:** Исхаков Дамир Айратович, группа 3823Б1ПР5
**Технологии:** SEQ-MPI. 
**Вариант:** '6'

## 1. Введение
Реализация передачи данных от одно процесса к другому путём линейной обработки. Проще говоря передача данных от процесса 0 данных в процесс 3 с помощью последовательной реализации, в которой информация сначала отправляется в процесс 1 (ближайший сосед, т.к 0 < 3  идём повозрастающей), после из процесса 1 в процесс 2, а уже потом из процесса 2 в процесс 3. Задача созранить данные сугубо в процессе, являющимся пунктом назначения, не сохраняя их в попутных процессах 
(0->1; 1->2; 2->3)

## 2. Постановка задачи
**Формальная задача**: Реализовать алгоритм передачи вектора целых чисел от заданного процесса-источника (head) к заданному процессу-приёмнику (tail) через цепочку промежуточных процессов в соответствии с линейной топологией.

**Входные данные**:
Структура Massege включающая в себя такие парраметры, как: 
 * head_process (int) — ранг процесса-источника

 * tail_process (int) — ранг процесса-приёмника

 * data (std::vector<int>) — вектор целых чисел для передачи (инициализируется только на процессе-источнике)

 * delivered (bool) — флаг, указывающий, были ли данные уже доставлены (входное значение всегда false) 

**Выходные данные**: 
Та же структура Massege только с добавлением параметра отвечающего за количество процессов

 * Message структура:

 *  head_process (int) — сохраняется из входных данных

 *  tail_process (int) — сохраняется из входных данных

 *  data (std::vector<int>) — вектор данных - на процессах head и tail: содержит переданные данные, на остальных процессах: пустой вектор

 *  delivered (bool) - на процессах head и tail: true, на остальных процессах: false

 * Количество процессов / processes_number (int) — общее число процессов в коммуникаторе MPI

**Ограничения**:

 * Индексы head_process и tail_process должны находиться в диапазоне [0, world_size-1]

 * На процессе-источнике вектор данных не должен быть пустым

 * Флаг delivered на входе всегда должен быть false

 * Если head_process == tail_process, данные остаются на том же процессе

## 3. Базовый алгоритм (Последовательный)

```cpp
const auto &input = GetInput();

  int head_process = input.head_process;
  int tail_process = input.tail_process;
  std::vector<int> local_data = input.data;
  bool delivered = true;

  Message result;
  result.head_process = head_process;
  result.tail_process = tail_process;
  result.data = local_data;
  result.delivered = delivered;

  GetOutput() = std::make_tuple(result, 1);

  return true;
```

## 4. Схема распараллеливания
### 4.1 Обработка исключений и нестандартных ситуаций

```cpp

  if (input.head_process < 0) {
    return false;
  }
  if (input.head_process >= world_size) {
    return false;
  }

  if (input.tail_process < 0) {
    return false;
  }
  if (input.tail_process >= world_size) {
    return false;
  }

  int is_valid_local = 1;

  if (world_rank == input.head_process) {
    if (input.data.empty() || input.delivered) {
      is_valid_local = 0;
    }
  }
```

``` cpp
 if (head_process == tail_process) {
    if (world_rank == head_process) {
      result.data = input.data;
      result.delivered = true;
    } else {
      result.data = {};
      result.delivered = false;
    }
    GetOutput() = std::make_pair(result, world_size);
    return true;
  }
```

### 4.2 Определение направления передачи данных

```cpp
  int direction;

  if (head_process < tail_process) {
    direction = 1;
  } else {
    direction = -1;
  }
```

### 4.3 Отбрасывание процессов, не учавствующих в передаче данных
Те процессы, которые либо вне Головы и Хвоста (пример: голова 1, хвост 3, процесс 0 отбрасывается, т.к не принимает участие в передаче данных)

``` cpp
  bool participate;
  if (direction > 0) {
    participate = ((world_rank >= head_process) && (world_rank <= tail_process));
  } else {
    participate = ((world_rank <= head_process) && (world_rank >= tail_process));
  }

  if (!participate) {
    result.data = {};
    result.delivered = false;
    GetOutput() = std::make_pair(result, world_size);
    return true;
  }

```

### 4.4 Передача данных

```cpp
if (is_head) {
    local_data = input.data;

    int local_data_size = static_cast<int>(local_data.size());
    MPI_Send(&local_data_size, 1, MPI_INT, next_process, 0, MPI_COMM_WORLD);
    MPI_Send(local_data.data(), local_data_size, MPI_INT, next_process, 1, MPI_COMM_WORLD);

    result.data = local_data;
    result.delivered = true;
  } else if (is_tail) {
    int local_data_size = 0;
    MPI_Recv(&local_data_size, 1, MPI_INT, previous_process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    local_data.resize(local_data_size);
    MPI_Recv(local_data.data(), local_data_size, MPI_INT, previous_process, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    result.data = std::move(local_data);
    result.delivered = true;
  } else {
    int local_data_size = 0;
    MPI_Recv(&local_data_size, 1, MPI_INT, previous_process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    local_data.resize(local_data_size);
    MPI_Recv(local_data.data(), local_data_size, MPI_INT, previous_process, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Send(&local_data_size, 1, MPI_INT, next_process, 0, MPI_COMM_WORLD);
    MPI_Send(local_data.data(), local_data_size, MPI_INT, next_process, 1, MPI_COMM_WORLD);

    result.data = {};
    result.delivered = false;
}
```

## 5. Детали реализации

### 5.1 Структуры данных
```cpp
struct Message {
  int head_process;
  int tail_process;
  std::vector<int> data;
  bool delivered;
};
```

### 5.2 Ключевые функции MPI

 * MPI_Comm_size, MPI_Comm_rank — получение информации о размере коммуникатора и ранге процесса

 * MPI_Send, MPI_Recv — отправка и приём данных

 * MPI_Allreduce — коллективная операция для синхронизации результатов валидации

 * MPI_Barrier — синхронизация процессов

## 6. Экспериментальная установка
- **Hardware/OS:** 
 * Процессор: 12th Gen Intel(R) Core(TM) i5-12500Hl
 * Ядра/Потоки: 12 ядер, 16 потоков
 * ОЗУ: 16GB DDR4 3200МГц
 * ОС: Linux Mint 22.2

- **Toolchain:** 
 * gcc --version 13.3.0
 * mpirun (Open MPI) 4.1.6
 * cmake version 3.28.3

- **Environment:** 
 * Количество процессов MPI (1, 2, 4)

- **Data:**  
 * Для функциональных тестов: вектора размером от 5 до 20

 * Для тестов производительности: вектора размером от 5000, до 10000

## 7. Результаты и обсуждение

### 7.1 Корректность
**Метод проверки**:
Реализованы функциональные тесты на базе Google Test Framework. Тесты проверяют:

  * Корректность передачи данных от head к tail процессу

  * Соответствие выходных данных входным на head и tail процессах

  * Пустые данные и флаг delivered = false на непосещающих процессах

  * Обработку граничных случаев (head == tail, невалидные индексы)

**Тестовые случаи**:

  * SingleProcess: head = tail = 0, проверка тривиального случая

  * TwoProcesses: head = 0, tail = 1, минимальная передача между двумя процессами

  * ThreeOrMoreProcesses: head = 0, tail = size-1, передача через все процессы

  * FourOrMoreProcesses: head = 0, tail = size-1, длинная цепочка передачи

**Проверка инвариантов**:

  * На head и tail процессах флаг delivered всегда true

  * На непосещающих процессах флаг delivered всегда false

  * Данные на head и tail процессах всегда совпадают с отправленными

  * Данные на промежуточных процессах всегда пусты

### 7.2 Производительность
Present time, speedup and efficiency. Example table:

| Mode  | Count    | Time, s   | Speedup   | Efficiency    |
|-------|----------|-----------|-----------|---------------|
| seq   | 1        | 0.6970    |   1.00    |     1.00      |
| mpi   | 2        | 1.1581    |   0.60    |     0.30      |
| mpi   | 4        | 1.8200    |   0.38    |     0.10      |

- Программа показала лишь замедление выполнения (0.38× на 4 процессах)
- Низкая эффективность (<20% на 4 процессах)

## 8. Выводы
Реализация MPI не обеспечило значительного ускорения

## 9. Ссылки
1. Документация по курсу - https://learning-process.github.io/parallel_programming_course/ru/common_information/report.html
2. Записи лекций - https://disk.yandex.ru/d/NvHFyhOJCQU65w
3. Гугл - https://www.google.com/
