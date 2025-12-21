# Линейка

**Студент:** Исхаков Дамир Айратович, группа 3823Б1ПР5
**Технологии:** SEQ-MPI. 
**Вариант:** '6'

## 1. Введение
Реализация передачи данных от одного процесса к другому путём линейной обработки. Проще говоря передача данных от процесса 0 данных в процесс 3 с помощью последовательной реализации, в которой информация сначала отправляется в процесс 1 (ближайший сосед, т.к 0 < 3  идём по возрастающей), после из процесса 1 в процесс 2, а уже потом из процесса 2 в процесс 3. Задача сохранить данные сугубо в процессе, являющемся пунктом назначения, не сохраняя их в попутных процессах 
(0->1; 1->2; 2->3)

## 2. Постановка задачи
**Формальная задача**: Реализовать алгоритм передачи вектора целых чисел от заданного процесса-источника (head) к заданному процессу-приёмнику (tail) через цепочку промежуточных процессов в соответствии с линейной топологией.

**Входные данные**:
Структура Message включающая в себя такие парраметры, как: 
 * head_process (int) — ранг процесса-источника

 * tail_process (int) — ранг процесса-приёмника

 * data (std::vector<int>) — вектор целых чисел для передачи (инициализируется только на процессе-источнике)

 * delivered (bool) — флаг, указывающий, были ли данные уже доставлены (входное значение всегда false) 

**Выходные данные**: 
Та же структура Message

 * Message структура:

 *  head_process (int) — сохраняется из входных данных

 *  tail_process (int) — сохраняется из входных данных

 *  data (std::vector<int>) — вектор данных - на процессах head и tail: содержит переданные данные, на остальных процессах: пустой вектор

 *  delivered (bool) - на процессе tail: true, на остальных процессах: false

**Ограничения**:

 * Индексы head_process и tail_process должны находиться в диапазоне [0, world_size-1]

 * На head_process вектор данных не должен быть пустым

 * Флаг delivered на входе всегда должен быть false

 * Если head_process == tail_process, данные остаются на том же процессе

## 3. Базовый алгоритм (Последовательный)

```cpp
bool IskhakovDLinearTopologySEQ::RunImpl() {
    const auto &input = GetInput();
    Message result;
    
    result.head_process = input.head_process;
    result.tail_process = input.tail_process;
    result.set_data(input.data);  
    result.delivered = true;    
    
    GetOutput() = result;
    return true;
}
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
```

``` cpp

  if (world_rank == input.head_process) {
    if (input.data.empty()) {
      is_valid_local = 0;
    }
    if (input.delivered) {
      is_valid_local = 0;
    }
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

### 4.3 Отбрасывание процессов, не участвующих в передаче данных
Те процессы, которые либо вне Головы и Хвоста (пример: голова 1, хвост 3, процесс 0 отбрасывается, т.к не принимает участие в передаче данных)

``` cpp
  bool participate;
  if (direction > 0) {
    participate = ((world_rank >= head_process) && (world_rank <= tail_process));
  } else {
    participate = ((world_rank <= head_process) && (world_rank >= tail_process));
  }

  if (!participate) {
    result.set_data({});
    result.delivered = false;
    GetOutput() = result;
    return true;
  }
```

### 4.4 Логика передачи данных

```cpp
if (is_head) {
    local_data = input.data;
    local_data_size = static_cast<int>(local_data.size());

    MPI_Request requests[2];
    MPI_Isend(&local_data_size, 1, MPI_INT, next_process, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(local_data.data(), local_data_size, MPI_INT, next_process, 1, MPI_COMM_WORLD, &requests[1]);
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

    result.set_data(local_data);
    result.delivered = false;
  } else if (is_tail) {
    MPI_Recv(&local_data_size, 1, MPI_INT, previous_process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    local_data.resize(local_data_size);
    MPI_Recv(local_data.data(), local_data_size, MPI_INT, previous_process, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    result.set_data(std::move(local_data));
    result.delivered = true;
  } else {
    MPI_Recv(&local_data_size, 1, MPI_INT, previous_process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    local_data.resize(local_data_size);
    MPI_Recv(local_data.data(), local_data_size, MPI_INT, previous_process, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Request requests[2];
    MPI_Isend(&local_data_size, 1, MPI_INT, next_process, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(local_data.data(), local_data_size, MPI_INT, next_process, 1, MPI_COMM_WORLD, &requests[1]);
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

    result.set_data({});
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

 * MPI_Isend, MPI_Waitall — неблокирующие операции для предотвращения deadlock

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
 * Для функциональных тестов: вектора размером от 5 до 70

 * "вектор из 25 000 000 элементов (100 MB данных)"

## 7. Результаты и обсуждение

### 7.1 Корректность
**Метод проверки**:

 * **SEQ тесты** (2 теста): проверка базовой функциональности на одном процессе
 * **MPI тесты** (14 тестов): проверка распределенной работы на 2+ процессах

**Проверяемые аспекты**:

* **Корректность передачи данных**: данные успешно доставляются от head к tail процессу
* **Семантика флага delivered**: 
  - tail процесс: delivered = true (данные получены)
  - head процесс: delivered = false (данные отправлены, подтверждение не получено)
  - промежуточные процессы: delivered = false (только передатчики)
  - непричастные процессы: delivered = false
* **Целостность данных**: 
  - tail процесс получает точную копию исходных данных
  - head процесс сохраняет исходные данные для отладки
  - промежуточные процессы не сохраняют данные
* **Граничные случаи**:
  - head == tail (один процесс)
  - head < tail (передача вперед)
  - head > tail (передача назад)
  - невалидные индексы
  - пустые данные на head процессе
  - already delivered данные

**Тестовые сценарии**:

1. **Базовые случаи** (head == tail):
   - 5 элементов на процессе 0
   - 10 элементов на процессе 0

2. **Передача между соседними процессами**:
   - 0 → 1 (15 элементов)
   - 1 → 0 (20 элементов)

3. **Передача через промежуточные процессы**:
   - 0 → 2 (25 элементов) через процесс 1
   - 2 → 0 (30 элементов) через процесс 1
   - 1 → 2 (35 элементов)
   - 2 → 1 (40 элементов)

4. **Длинные цепочки передачи**:
   - 0 → 3 (45 элементов) через процессы 1, 2
   - 3 → 0 (50 элементов) через процессы 2, 1
   - 1 → 3 (55 элементов) через процесс 2
   - 3 → 1 (60 элементов) через процесс 2
   - 2 → 3 (65 элементов)
   - 3 → 2 (70 элементов)

### 7.2 Производительность
Результаты тестов производительности для вектора из 25 000 000 целых чисел:

| Mode  | Процессов | Время, с   | Speedup  | Efficiency |
|-------|-----------|------------|----------|------------|
| seq   | 1         | 0.0790     | 1.00     | 1.00       |
| mpi   | 2         | 0.2556     | 0.31     | 0.15       |
| mpi   | 4         | 0.3425     | 0.23     | 0.06       |


- Программа показала лишь замедление выполнения (0.23х на 4 процессах)
- Низкая эффективность (<20% на 4 процессах)

## 8. Выводы
Реализация MPI не обеспечило значительного ускорения

## 9. Ссылки
1. Документация по курсу - https://learning-process.github.io/parallel_programming_course/ru/common_information/report.html
2. Записи лекций - https://disk.yandex.ru/d/NvHFyhOJCQU65w
3. Гугл - https://www.google.com/
