# Построение выпуклой оболочки – проход Грэхема

**Студент:** Исхаков Дамир Айратович, группа 3823Б1ПР5
**Технологии:** SEQ-MPI. 
**Вариант:** '24'

## 1. Введение
Был реализован алгоритм Грэхема для построения выпуклой оболочки множества точек. Алгоритм работает в три этапа:

   1. Нахождение точки с минимальной y-координатой

   2. Сортировка точек по полярному углу

   3.  Построение оболочки через проверку векторного произведения

Особенность реализации — создание как последовательной (SEQ), так и параллельной (MPI) версии с использованием бинарного дерева для слияния частичных результатов. Частные результаты получаются путём разбития точек отночительно процессов (минимум 10 для каждого), после каждый процесс строит свои оболочки, а в конце в 0 процессе они соединяются воедино. Цель работы — сравнение производительности двух подходов.

## 2. Постановка задачи
**Формальная задача**: Построить выпуклую оболочку множества точек на плоскости с использованием алгоритма Грэхема в последовательной и параллельной реализации

**Входные данные**:
 * std::vector<Point> - вектор точек, где Point имеет координаты x и y (тип double)

**Выходные данные**: 
 * std::vector<Point> - точки выпуклой оболочки в порядке обхода против часовой стрелки

## 3. Базовый алгоритм (Последовательный)

```cpp
bool IskhakovDGrahamConvexHullSEQ::RunImpl() {
  std::vector<Point> points = GetInput();
  
  size_t index_min_point = 0;
  for (size_t i = 1; i < points.size(); i++) {
    if (points[i].y < points[index_min_point].y ||
        (points[i].y == points[index_min_point].y && 
         points[i].x < points[index_min_point].x)) {
      index_min_point = i;
    }
  }
  
  std::swap(points[0], points[index_min_point]);
  Point start_point = points[0];
  
  std::sort(points.begin() + 1, points.end(),
            [&start_point](const Point& a, const Point& b) {
              double orient = (a.x - start_point.x) * (b.y - start_point.y) -
                             (a.y - start_point.y) * (b.x - start_point.x);
              return orient > 0;
            });
  
  std::vector<Point> hull;
  hull.push_back(points[0]);
  hull.push_back(points[1]);
  
  for (size_t i = 2; i < points.size(); i++) {
    while (hull.size() >= 2) {
      Point& a = hull[hull.size() - 2];
      Point& b = hull[hull.size() - 1];
      double orient = (b.x - a.x) * (points[i].y - a.y) -
                     (b.y - a.y) * (points[i].x - a.x);
      if (orient <= 0) hull.pop_back();
      else break;
    }
    hull.push_back(points[i]);
  }
  
  GetOutput() = hull;
  return true;
}
```

## 4. Схема распараллеливания
**Общая стратегия:** Разделение данных, независимое построение частичных оболочек (Graham Scan) на каждом процессе, и последующее слияние результатов в бинарном дереве с использованием алгоритма слияния двух выпуклых оболочек.

Логика адаптивного выбора процессов:

    Менее 10 точек → 1 процесс (последовательная версия)

    10-19 точек → 1 процесс (степень двойки ≤ 1)

    20-39 точек → 2 процесса (степень двойки ≤ 2, минимум 10 точек на процесс)

    40-79 точек → 4 процесса (степень двойки ≤ 4)

    80+ точек → 8 процессов (если доступно) и т.д.

Это гарантирует, что на каждый процесс будет достаточно точек для эффективной работы и избегает проблем с малым количеством точек на процесс.

### 4.1 Подготовка и распределение данных

```cpp
std::vector<Point> PrepareAndDistributeData(int world_rank, int world_size) {

  int optimal_active_procs = world_size;
  while (optimal_active_procs > 1 && points_count < 3 * optimal_active_procs) {
    optimal_active_procs--;
  }
  

  std::vector<double> all_x, all_y;
  if (world_rank == 0) {

  }
  

  MPI_Scatterv(all_x.data(), proc_points_counts.data(), displacements.data(),
               MPI_DOUBLE, local_x.data(), my_count_points, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
  
  return local_points;
}
```

### 4.2 Построение локальных оболочек

Каждый процесс выполняет последовательный алгоритм Грэхема на своем наборе точек. Результат — локальная выпуклая оболочка.

### 4.3 Слияние оболочек в бинарном дереве

Функция MergeHullsBinaryTree реализует алгоритм слияния двух выпуклых многоугольников, обходя их по часовой стрелке и используя технику "двух указателей".

```cpp
std::vector<Point> MergeHullsBinaryTree(int world_rank, int world_size, 
                                        const std::vector<Point>& local_hull) {
  std::vector<Point> current_hull = local_hull;
  

  int active_procs = 1;
  while (active_procs * 2 <= world_size) {
    active_procs *= 2;
  }
  

  for (int step = 1; step < active_procs; step *= 2) {
    int partner = world_rank ^ step;
    
    if (partner < active_procs) {

      int my_hull_size = current_hull.size();
      int partner_hull_size;
      MPI_Sendrecv(&my_hull_size, 1, MPI_INT, partner, 0,
                   &partner_hull_size, 1, MPI_INT, partner, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
      if (my_hull_size > 0 && partner_hull_size > 0) {
        
        current_hull = MergeHulls(current_hull, remote_hull);
      }
    }
  }
  
  return current_hull;
}
```

## 5. Детали реализации

### 5.1 Структуры данных
```cpp
struct Point {
  double x, y;
  
  Point(double x = 0, double y = 0) : x(x), y(y) {}
  
  bool operator<(const Point& other) const {
    return std::tie(y, x) < std::tie(other.y, other.x);
  }
  
  bool operator==(const Point& other) const {
    static constexpr double kEpsilon = 1e-9;
    return std::abs(x - other.x) < kEpsilon && 
           std::abs(y - other.y) < kEpsilon;
  }
};
```

### 5.2 Ключевые функции MPI

 * MPI_Comm_size, MPI_Comm_rank - получение информации о коммуникаторе
 * MPI_Bcast - широковещательная рассылка
 * MPI_Scatterv - распределение данных с переменным размером
 * MPI_Gather - сбор данных
 * MPI_Sendrecv - двусторонний обмен данными
 * MPI_Barrier - синхронизация процессов

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
 * Количество точек: 1,000,000 со случайными значеними координат в диапазоне [0, 10000]
 * Для функциональных тестов были использованы заранее просчитанные тачки

## 7. Результаты и обсуждение

### 7.1 Корректность

**Метод проверки:**

    SEQ тесты: проверка базовой функциональности на одном процессе

    MPI тесты: проверка распределенной работы на 2+ процессах

**Проверяемые аспекты:**

    Корректность алгоритма: выпуклая оболочка строится правильно

    Обработка граничных случаев:

        Меньше 3 точек

        Коллинеарные точки

        Все точки на одной вертикальной/горизонтальной линии

    Согласованность результатов: SEQ и MPI версии дают одинаковый результат

    Эффективность распределения: данные корректно распределяются между процессами

### 7.2 Производительность

Результаты тестов производительности для 1,000,000 точек:

| Mode  | Процессов | Время, с   | Speedup  | Efficiency |
|-------|-----------|------------|----------|------------|
| seq   | 1         | 0.216      | 1.00     | 1.00       |
| mpi   | 2         | 0.137      | 1.58     | 0.79       |
| mpi   | 4         | 0.090      | 2.40     | 0.60       |


- Программа показала значительное ускорение (на 2 процессах: ускорение 1.58×, на 4 процессах: ускорение 2.4×) и хорошую эфективность (на 2 процессах: 79% (высокая эффективность), на 4 процессах: 60% (приемлемая эффективность))

## 8. Выводы
Алгоритм Грэхема хорошо поддается параллелизации с использованием стратегии "разделяй и властвуй". Достигнуто ускорение 2.4× на 4 процессах.

## 9. Ссылки
1. Документация по курсу - https://learning-process.github.io/parallel_programming_course/ru/common_information/report.html
2. Записи лекций - https://disk.yandex.ru/d/NvHFyhOJCQU65w
3. Гугл - https://www.google.com/