import data.DataReader;
import data.Image;

import java.util.List;

public class Main {
    public static void main(String[] args)
    {
        String dir = System.getProperty("user.dir");

        List<Image> images = new DataReader().readData(dir+"/downloads/data/mnist_test.csv");

        System.out.printf(images.getFirst().toString());
    }
}
