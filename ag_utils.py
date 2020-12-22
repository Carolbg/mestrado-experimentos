from random import uniform, randint

def randomFloat(start, end):
    value = uniform(start, end)
    # print('value in randomFloat', value)
    return value

def randomInt(start, end):
    value = randint(start, end)
    # print('value in randomInt', value)
    return value
