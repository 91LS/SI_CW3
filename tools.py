import classes
import math


# Functions for read rule based system. <-------------------------------------------------------------------------------
def get_system_objects(system_file):
    """Return numpy array with integers that represent Rule Based System and dictionary to encode true values."""
    system_file.seek(0)  # return to beginning of file
    objects = []
    for line in system_file:
        system_object = line.strip().split(' ')
        float_object = list(map(float, system_object[:-1]))
        objects.append(classes.Distance(float_object, system_object[-1]))
    return objects


def classify_objects(measure, trn_system, tst_system, k):
    """Classify tst_system by trn_system."""
    for tst_object in tst_system:
        classify_object(measure, tst_object, trn_system, k)


def classify_object(measure, tst_object, trn_system, k):
    """Return calculated distance between objects with decision label."""
    distances = {}
    for trn_object in trn_system:
        if trn_object.decision not in distances:
            distances[trn_object.decision] = [measure(trn_object.descriptors, tst_object.descriptors)]
        else:
            distances[trn_object.decision].append(measure(trn_object.descriptors, tst_object.descriptors))
    for key, value in distances.items():
        sorted_distances = sorted(distances.get(key))
        distances[key] = sum(sorted_distances[:k])

    minimum_pair = get_minimum_pair(distances)
    if is_value_unique(distances, minimum_pair[1]):
        tst_object.set_classifier_decision(minimum_pair[0])


def get_minimum_pair(distances):
    """Return the key from dictionary with minimal value."""
    minimum_key = list(distances.keys())[0]
    minimum_value = distances.get(minimum_key)
    minimum_pair = [minimum_key, minimum_value]

    for key, value in distances.items():
        if value < minimum_value:
            minimum_pair[0] = key
            minimum_pair[1] = value
    return minimum_pair


def is_value_unique(distances, minimum_value):
    counter = 0
    for value in distances.values():
        if value == minimum_value:
            counter += 1
    if counter > 1:
        return False
    return True


#  Metrics. <-----------------------------------------------------------------------------------------------------------
def get_canberra(first, second):
    """Return distance calculated by Canberra metric."""
    distance = [math.fabs((x - y)/(x + y)) for x, y in zip(first, second)]
    distance = math.sqrt(sum(distance))
    return distance


def get_chebyshev(first, second):
    """Return distance calculated by Chebyshev metric."""
    distance = [math.fabs((x - y)) for x, y in zip(first, second)]
    distance = max(distance)
    return distance


def get_euclidean(first, second):
    """Return distance calculated by euclidean metric."""
    distance = [(x - y) ** 2 for x, y in zip(first, second)]
    distance = math.sqrt(sum(distance))
    return distance


def get_manhattan(first, second):
    """Return distance calculated by Manhattan metric."""
    distance = [math.fabs(x - y) for x, y in zip(first, second)]
    distance = sum(distance)
    return distance


def get_pearson(first, second):
    """Return distance calculated by Pearson correlation coefficient."""
    x_avg = sum(first)/len(first)
    y_avg = sum(second)/len(second)

    x_denominator = math.sqrt(sum((x - x_avg) ** 2 for x in first)/len(first))
    y_denominator = math.sqrt(sum((y - y_avg) ** 2 for y in second) / len(second))

    distance = [((x - x_avg)/x_denominator)*((y - y_avg)/y_denominator) for x, y in zip(first, second)]
    distance = sum(distance)/len(first)
    return 1 - math.fabs(distance)


#  Other helpful functions. <-------------------------------------------------------------------------------------------
def get_maximum_k_size(trn_system):
    """Return lowest number of objects from class."""
    classes_count = {}
    for trn_object in trn_system:
        if trn_object.decision not in classes_count:
            classes_count[trn_object.decision] = 1
        else:
            classes_count[trn_object.decision] += 1
    return min(classes_count.values())


def get_values(tst_class, trn_classes, tst_system):
    """Return row for prediction matrix."""
    row = get_row_init(trn_classes)
    class_objects = get_object_classes(tst_class, tst_system)

    for class_object in class_objects:
        if class_object.classifier_decision is not None:
            row[class_object.classifier_decision] += 1

    row_list = []
    for trn_class in trn_classes:
        row_list.append(row.get(trn_class))
    data = get_data(class_objects, tst_system)
    row_list += data
    return row_list


def get_row_init(trn_classes):
    """Initialize dictionary with 0 value."""
    row = {}
    for trn_class in trn_classes:
        row[trn_class] = 0
    return row


def get_object_classes(tst_class, tst_system):
    """Return list of class object."""
    return [class_object for class_object in tst_system if class_object.decision == tst_class]


def get_classes(system):
    """Return unique classes from system."""
    unique = []
    for decision_object in system:
        if decision_object.decision not in unique:
            unique.append(decision_object.decision)
    return sorted(unique)


def get_data(class_objects, tst_system):
    """Return: number of class objects, accuracy, cover and tru positive rate."""
    class_length = class_objects.__len__()
    grabbed = 0
    correct = 0
    wrong = 0

    class_decision = class_objects[0].decision
    for class_object in class_objects:
        if class_object.classifier_decision is not None:
            grabbed += 1
            if class_object.classifier_decision == class_object.decision:
                correct += 1
    acc = correct/grabbed if grabbed != 0 else "No one is grabbed"
    cov = grabbed/class_length

    others_objects = [other for other in tst_system if other not in class_objects]
    for other in others_objects:
        if other.classifier_decision == class_decision:
            wrong += 1
    tpr = correct/(correct + wrong)

    return [class_length, acc, cov, tpr]


def transform_last_column_to_row(matrix):
    """Remove last column from matrix and add it as row."""
    column = []
    for row in matrix:
        column.append(row.pop())
    matrix.append(column)


def get_global(tst_system):
    """Return accuracy and cover for all TST system."""
    correct = 0
    grabbed = 0

    for tst_object in tst_system:
        if tst_object.classifier_decision is not None:
            grabbed += 1
            if tst_object.classifier_decision == tst_object.decision:
                correct += 1

    acc = correct/grabbed if grabbed != 0 else "No one is grabbed"
    cov = grabbed/tst_system.__len__()
    return acc, cov
