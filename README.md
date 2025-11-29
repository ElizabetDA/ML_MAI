# HW2 — Decision Trees (МАИ, ФИИТ)

Домашнее задание №2 по курсу машинного обучения: решающие деревья.

**Автор:** Дьяченко Елизавета, М8О-312Б-23  

---

## Содержание репозитория

- `ml_hw2.ipynb` — основной ноутбук с выполненным заданием:
  - визуализация разделяющих поверхностей для `DecisionTreeClassifier`;
  - исследование влияния гиперпараметров (`max_depth`, `min_samples_leaf`);
  - анализ признаков по критерию Джини на `students.csv`;
  - эксперименты с несколькими датасетами (mushrooms, tic-tac-toe, cars, nursery);
  - сравнение собственного дерева с `DecisionTreeClassifier` из `sklearn`.

- `ml.py` — реализация собственного решающего дерева:
  - функция `find_best_split` (критерий Джини);
  - класс `DecisionTree` с поддержкой `real` и `categorical` признаков;
  - параметры `max_depth`, `min_samples_split`, `min_samples_leaf`.

- `students.csv`, `agaricus-lepiota.data`, `tic-tac-toe-endgame.csv`, `car.data`, `nursery.data` — используемые датасеты.

---

## Запуск

1. Открыть `ml_hw2.ipynb` в Jupyter / Google Colab.
2. Убедиться, что в той же директории лежат:
   - `ml.py`
   - все необходимые датасеты.
3. Последовательно выполнить все ячейки ноутбука.

---
