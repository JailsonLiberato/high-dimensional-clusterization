class FileUtil:

    @staticmethod
    def create_file(array_data, array_metrics_names, filename):
        file = open("results/" + filename, "w+")
        counter = 0
        for name in array_metrics_names:
            file.write(name + ":")
            for value in array_data[counter]:
                file.write("%f|" % value)
            file.write("\n")
            counter += 1
        file.close()

    @staticmethod
    def read_file(filename):
        file = open("results/" + filename, "r")
        if file.mode == 'r':
            contents = file.read()
            array_content = contents.split("\n")
            array_content.pop()
        return array_content
