import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.Map;

import static org.apache.spark.sql.functions.col;

public class HousingPriceAnalysis {

    public static void main(String[] args) {

        String path = "src/main/resources/kc_house_data.csv";
        Dataset<Row> dataset = Util.getSession().read()
                .option("header", true)
                .option("inferSchema", true) // since we need only numeric types
                .csv(path);

        // prepare the vector of features
        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[]{"bedrooms", "bathrooms", "sqft_living", "sqft_lot","floors","grade"});
        vectorAssembler.setOutputCol("features");

        Dataset<Row> inputData = vectorAssembler.transform(dataset)
                .select(col("price"), col("features")) // select only required features
                .withColumnsRenamed(Map.of("price","label"));

        // split into 80% training data and 20% testing data
        Dataset<Row> [] sets = inputData.randomSplit(new double[] {0.8, 0.2});

        Dataset<Row> trainingData = sets[0];
        Dataset<Row> testingData = sets[1];

        LinearRegression linearRegression = new LinearRegression();
        // feed the training data to the model
        LinearRegressionModel model = linearRegression.fit(trainingData);

        double r2Training = model.summary().r2();
        double rmseTraining = model.summary().rootMeanSquaredError();
        System.out.println("R2 training : " + r2Training + " | RMSE training : " + rmseTraining);

        // test the accuracy by passing the test data to the model
        model.transform(testingData).show();

        // evaluate accuracy
        // R square -> between 0 and 1, bigger the better
        // RMSE -> smaller the better

        double r2Test = model.evaluate(testingData).r2();
        double rmseTest = model.evaluate(testingData).rootMeanSquaredError();
        System.out.println("R2 testing : " + r2Training + " | RMSE testing : " + rmseTraining);
    }
}
