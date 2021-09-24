with open('experiments_1407.csv', 'r', newline='') as read_obj:
    import csv
    f = csv.reader(read_obj)
    new_file = list()
    for r in f:
        #if r[0] == 'date':
        #    r.insert(12, 'only_next_frame')
        #
        if len(r) != 16:
            r.insert(12, '-')
        
        new_file.append(r)
        
        #if r[7] != '-' and r[7] != 'frame dist thresh 2':
        #    pass
        #else:
        #    new_file.append(r)

for r in new_file:
    with open('experiment_n.csv', 'a+', newline='') as write_obj:
        f = csv.writer(write_obj)
        f.writerow(r)
