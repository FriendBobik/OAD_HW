import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

# Определяем структуру для листьев и узлов дерева
Leaf = namedtuple("Leaf", ("value",))
Node = namedtuple("Node", ("feature", "value", "left", "right"))

# Класс для построения дерева решений
class DecisionTree:
    def __init__(self, max_depth=np.inf):
        # Инициализируем максимальную глубину дерева и корень
        self.max_depth = max_depth
        self.root = None
        self.classes = None

    def fit(self, X, y):
        # Преобразуем входные данные к нужному формату
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        # Сохраняем уникальные классы
        self.classes = np.unique(y)
        # Строим дерево
        self.root = self._build_tree(X, y)
        return self

    def partition(self, X, y, feature, value):
        # Разделяем данные на две части по указанному признаку и значению
        right = X[:, feature] >= value
        left = np.logical_not(right)

        return (X[left], y[left]), (X[right], y[right])

    def _build_tree(self, X, y, depth=1):
        # Строим дерево рекурсивно, пока не достигнем максимальной глубины или если в узле меньше 2 элементов
        if depth > self.max_depth or X.shape[0] < 2:
            return Leaf(self.leaf_value(y))

        # Ищем лучший признак и значение для разбиения
        feature, value = self.find_split(X, y)
        xy_left, xy_right = self.partition(X, y, feature, value)

        # Рекурсивно строим левую и правую поддеревья
        left = self._build_tree(*xy_left, depth=depth+1)
        right = self._build_tree(*xy_right, depth=depth+1)

        return Node(feature, value, left, right)

    def find_split(self, X, y):
        # Выбираем случайную половину признаков для поиска лучшего разбиения
        features = np.random.choice(np.arange(X.shape[1]), X.shape[1] // 2)

        best_feature = None
        best_value = None
        best_impurity = np.inf

        for feature in features:
            # Сортируем значения признака
            order = np.argsort(X[:, feature])
            impurities = np.empty(order.shape[0] - 1)
            for split_index in np.arange(1, order.shape[0]):
                # Считаем нечистоту (импьюрити) для каждого разбиения
                left = order[:split_index]
                right = order[split_index:]
                impurities[split_index - 1] = self.impurity(y[left], y[right])

            # Находим минимальную нечистоту
            current_impurity = np.min(impurities)
            if current_impurity < best_impurity:
                best_impurity = current_impurity
                best_index = np.argmin(impurities) + 1
                best_feature = feature
                best_value = X[order[best_index], feature]

        return best_feature, best_value

    def impurity(self, y_left, y_right):
        # Считаем взвешенную сумму критериев для левой и правой частей разбиения
        left_count = y_left.shape[0]
        right_count = y_right.shape[0]
        count = left_count + right_count

        left_c = self.criteria(y_left)
        right_c = self.criteria(y_right)

        return (left_count / count * left_c + right_count / count * right_c)

    def criteria(self, y):
        # Используем дисперсию как критерий
        return np.var(y)

    def leaf_value(self, y):
        # Возвращаем среднее значение в листе
        if len(y) == 0:
            return 0  # или другое значение по умолчанию, если нужно
        return np.mean(y)

    def _predict_one(self, x):
        # Прогнозируем значение для одного элемента
        node = self.root
        while not isinstance(node, Leaf):
            if x[node.feature] >= node.value:
                node = node.right
            else:
                node = node.left
        return node.value

    def predict(self, X):
        # Прогнозируем значения для набора данных
        X = np.atleast_2d(X)
        y = np.empty(X.shape[0])
        for i, x in enumerate(X):
            y[i] = self._predict_one(x)
        return y

# Класс для реализации метода бэггинга
class Bagging:
    def __init__(self, n_estimators=100, max_depth=np.inf):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = []

    def fit(self, X, y):
        for i in range(self.n_estimators):
            # Выбираем случайную подвыборку из данных
            subset = np.random.choice(np.arange(X.shape[0]), X.shape[0] // 2)
            X0 = X[subset]
            y0 = y[subset]
            # Обучаем базовый алгоритм (дерево решений)
            estimator = DecisionTree(max_depth=self.max_depth)
            estimator.fit(X0, y0)
            self.estimators.append(estimator)
        return self

    def predict(self, X):
        # Прогнозируем значения для набора данных с использованием всех базовых алгоритмов и усредняем результат
        y = np.asarray([estimator.predict(X) for estimator in self.estimators])
        return np.mean(y, axis=0)

# Создаем набор данных
COLORS = np.asarray([[0, 1, 0], [1, 0, 1]])

X = np.random.normal(0, 1, size=(100, 2))
y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int)
X = np.concatenate([X, np.random.normal(0, 1, size=(50, 2))])
y = np.concatenate([y, np.random.choice([0, 1], 50)])

# Генерируем тестовые данные
X_test = np.random.normal(0, 1, size=(100, 2))
dtc = Bagging(max_depth=3, n_estimators=200).fit(X, y)

# Создаем сетку для визуализации
x_grid = np.linspace(-5, 5, 200)
xx, yy = np.meshgrid(x_grid, x_grid)
X_test = np.stack([xx, yy], axis=-1).reshape(-1, 2)
pred = dtc.predict(X_test).reshape(xx.shape)

# Визуализируем результаты
plt.contourf(xx, yy, pred, cmap="pink_r", alpha=0.5)
plt.colorbar()
# plt.scatter(*X.T, color=COLORS[y], alpha=0.5)  # Можно раскомментировать, чтобы отобразить исходные данные
plt.show()