import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class App {

    private static final Logger log = LoggerFactory.getLogger(App.class);

    public static void main(String[] args) {

        // create the context
        SparkConf conf = Util.getConfig();
        JavaSparkContext sparkContext = new JavaSparkContext(conf);

        List<Double> nums = new ArrayList<>();
        nums.add(3425.89);
        nums.add(425.89);
        nums.add(345.89);

        exploreCount(nums, sparkContext);
        exploreReduce(nums, sparkContext);
        exploreMap(List.of(2, 35, 46), sparkContext);
        exploreCount(nums, sparkContext);
        exploreTuple(List.of(2, 35, 46), sparkContext);
        explorePairRdd(sparkContext);
        exploreReadFromDisk(sparkContext);

        sparkContext.close();

    }

    /**
     * reads a file from hard disk or AWS s3 bucket or HDFS
     * @param sparkContext the context
     */
    private static void exploreReadFromDisk(JavaSparkContext sparkContext) {

        JavaRDD<String> fileRdd = sparkContext.textFile("src/main/resources/input.txt");

        FlatMapFunction<String, String> flatMapFunction = s -> {
            List<String> wordList = Arrays.asList(s.split(" "));
            return wordList.iterator();
        };

        fileRdd.flatMap(flatMapFunction)
                .collect()
                .forEach(System.out::println);

    }

    private static void explorePairRdd(JavaSparkContext sparkContext) {

        List<String> originalLogs = new ArrayList<>();
        originalLogs.add("INFO : Tuple (2,1.4142135623730951)");
        originalLogs.add("INFO : Count is 3");

        JavaRDD<String> rdd = sparkContext.parallelize(originalLogs);

        PairFunction<String, String, String> pf = s -> {
            var level = s.split(":")[0];
            var content = s.split(":")[1];
            return new Tuple2<>(level, content);
        };

        JavaPairRDD<String, String> pairRDD = rdd.mapToPair(pf);

        // extracting keys and values
        JavaRDD<String> keysRdd = pairRDD.keys();
        JavaRDD<String> vaulesRdd = pairRDD.values();

    }

    /**
     * Simple count operation
     *
     * @param nums         the dataset to operate on
     * @param sparkContext the context
     */
    private static void exploreCount(List<Double> nums, JavaSparkContext sparkContext) {
        JavaRDD<Double> rdd = sparkContext.parallelize(nums);
        int count = rdd.map(n -> 1)
                .reduce((a, b) -> a + b);
        log.info("Count is : {}", count);
    }

    /**
     * Simple reduce operation, demonstrating addition of elements
     *
     * @param nums         the dataset to operate on
     * @param sparkContext the context
     */
    private static void exploreReduce(List<Double> nums, JavaSparkContext sparkContext) {
        // build a RDD
        JavaRDD<Double> rdd = sparkContext.parallelize(nums);
        Function2<Double, Double, Double> f = (a, b) -> a + b;
        Double ans = rdd.reduce(f);
        log.info("Reduced : {}", ans);
    }

    /**
     * Simple reduce operation, demonstrating addition of elements
     *
     * @param nums         the dataset to operate on
     * @param sparkContext the context
     */
    private static void exploreMap(List<Integer> nums, JavaSparkContext sparkContext) {
        // build a RDD
        JavaRDD<Integer> rdd = sparkContext.parallelize(nums);
        rdd.map(Math::sqrt).collect().forEach(e -> log.info("Square Root : {}", e));
    }


    private static void exploreTuple(List<Integer> nums, JavaSparkContext sparkContext) {
        JavaRDD<Integer> rdd = sparkContext.parallelize(nums);
        JavaRDD<Tuple2<Integer, Double>> rdd2 = rdd.map(i -> new Tuple2<>(i, Math.sqrt(i)));
        rdd2.collect().forEach(t -> log.info("Tuple : {}", t));
    }
}
