
from collections import OrderedDict

nav_actions = {'MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown'}


def milestone_checker(targets, objects, actions):
    '''
    Check milestone for the batch
    actions = list of single previous actions
    targets length == objects_length
    '''
    assert len(targets) == len(objects)

    batch_milestone_reached = []
    for tgt, objs, acts in zip(targets, objects, actions):
        milestone_reached = False
        for tgt_type, tgts in tgt.items():
            for target in tgts:
                for obj in objs:
                    if target.casefold() in obj.casefold() and objs[obj] == 1:
                        milestone_reached = True
                if tgt_type == 'inter':
                    if acts["action"] not in nav_actions and target.casefold() in acts["arg"].casefold():
                        milestone_reached = True
                    else:
                        milestone_reached = False
        batch_milestone_reached.append(milestone_reached)

    return batch_milestone_reached        



if __name__ == '__main__':

    objects = [{'Pillow': 1, 'SideTable': 1, 'CreditCard': 1, 'CoffeeTable': 1, 'HandTowel': 1, 'Bed': 1, 'Cup': 1, 'Cart': 1, \
                'BreadSliced': 1, 'Fridge': 1, 'WineBottle': 0, 'ArmChair': 0}, {'Sink': 1, 'Painting': 1, 'Bread': 1, 'SinkBasin': 1, \
                'ShowerGlass': 1, 'Cabinet': 1, 'Pillow': 1, 'Pot': 0, 'Lettuce': 1, 'Toaster': 0, 'CellPhone': 0, 'Cloth': 1, 'WineBottle': 1, \
                'Cup': 1, 'PepperShaker': 0, 'CoffeeTable': 0, 'HandTowel': 1, 'Ladle': 0, 'Bowl': 0, 'Fork': 1, 'ScrubBrush': 1, 'Fridge': 1, \
                'Potato': 1, 'DeskLamp': 0, 'Kettle': 0, 'Knife': 0, 'Boots': 1}]
    
    targets = [OrderedDict([('nav', ['Fridge'])]), OrderedDict([('inter', ['cup'])])]

    actions = [{'action': "RotateLeft", 'arg': None}, {'action': "PickupObject", 'arg' : 'Cup'}]

    pred = milestone_checker(targets, objects, actions)
    print(pred)