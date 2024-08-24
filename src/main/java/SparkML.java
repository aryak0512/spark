import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Map;

import static org.apache.spark.sql.functions.col;

public class SparkML {

    public static void main(String[] args) {
        var session = Util.getSession();

        //prepareData(session);
        exploreLinearRegression(session);
    }

    private static void exploreLinearRegression(SparkSession session) {
        Dataset<Row> inputData = prepareData(session);
        LinearRegression linearRegression = new LinearRegression();
        LinearRegressionModel model = linearRegression.fit(inputData);
        var intercept = model.intercept();
        var coefficients = model.coefficients();

        System.out.println("Intercept : " + intercept + " Coefficients : " + coefficients);
        model.transform(inputData).show();
    }

    /**
     * prepare the right input data for model
     * <p>
     * The format is label and array of features. Array of features is actually a spark data structure called vector
     * </p>
     *
     * @param session
     * @return
     */
    private static Dataset<Row> prepareData(SparkSession session) {

        String path = "src/main/resources/GymCompetition.csv";
        Dataset<Row> dataset = session.read()
                .option("header", true)
                .option("inferSchema", true) // since we need only numeric types
                .csv(path);

        // prepare the vector of features
        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[]{"Age", "Height", "Weight"});
        vectorAssembler.setOutputCol("features");

        Dataset<Row> inputData = vectorAssembler.transform(dataset)
                .select(col("NoOfReps"), col("features")) // select only required features
                .withColumnsRenamed(Map.of("NoOfReps","label"));

        return inputData;
    }
}
