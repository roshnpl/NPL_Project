#-*-coding:UTF-8-*-

def main():
    f = open("debug_log.csv", "r")
    fw = open("bad_data.csv", "w")
    x_list = []
    y_list = []
    q_list = []

    while True:
        line = f.readline()
        if not line:
            break
        arr = line.split(",")
        x_arr = arr[0:10]
        x_str = ','.join(x_arr)
        if arr[7] == "None":
            y_arr = arr[10:13]
            y_str = ','.join(y_arr)
            q_arr = arr[13:]
            q_str = ','.join(q_arr)
        else:
            y_arr = arr[10:15]
            y_str = ','.join(y_arr)
            q_arr = arr[15:]
            q_str = ','.join(q_arr)
        if x_str in x_list:
            index = x_list.index(x_str)
            if y_list[index] != y_str:
                fw.write(x_list[index] + "," + y_list[index] + "," + q_list[index])
                fw.write(line)
        x_list.append(x_str)
        y_list.append(y_str)
        q_list.append(q_str)
    print("over")
    f.close()
    fw.close()

if __name__ == "__main__":
    main()
