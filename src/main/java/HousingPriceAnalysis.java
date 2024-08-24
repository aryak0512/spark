import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
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

        // split into 80% training & testing data and 20% holdOut data
        Dataset<Row> [] sets = inputData.randomSplit(new double[] {0.8, 0.2});

        Dataset<Row> trainingAndTestingData = sets[0];  // test data will be for tuning the values of regParam and elasticNetParam
        Dataset<Row> holdOutData = sets[1];             // for independent validation

        LinearRegression linearRegression = new LinearRegression();

        // adding the regParam and elasticNetParam params
        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
        // these are some sophisticated values we add for tuning the accuracy of the model
        ParamMap[] paramMap = paramGridBuilder
                .addGrid(linearRegression.regParam(), new double[]{0.01, 0.1, 0.5})
                .addGrid(linearRegression.elasticNetParam(), new double[]{0, 0.5, 1.0})
                .build();

        TrainValidationSplit split = new TrainValidationSplit()
                .setTrainRatio(0.8)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMap)
                .setEstimator(linearRegression);

        // feed the training data to the model
        TrainValidationSplitModel splitModel = split.fit(trainingAndTestingData);

        // pick the best model
        LinearRegressionModel model = (LinearRegressionModel) splitModel.bestModel();

        double r2Training = model.summary().r2();
        double rmseTraining = model.summary().rootMeanSquaredError();
        System.out.println("R2 training : " + r2Training + " | RMSE training : " + rmseTraining);

        // test the accuracy by passing the test data to the model
        model.transform(holdOutData).show();

        // evaluate accuracy
        // R square -> between 0 and 1, bigger the better
        // RMSE -> smaller the better

        double r2Test = model.evaluate(holdOutData).r2();
        double rmseTest = model.evaluate(holdOutData).rootMeanSquaredError();
        System.out.println("R2 testing : " + r2Test + " | RMSE testing : " + rmseTest);
    }
}
