import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.Map;

import static org.apache.spark.sql.functions.col;

/**
 * class that deals with non-numeric data types.
 * Such fields undergo indexing and hot encoding & we finally create vectors of these fields
 */
public class IndexingAndEncoding {

    public static void main(String[] args) {

        String path = "src/main/resources/GymCompetition.csv";
        Dataset<Row> inputData = Util.getSession().read()
                .option("header", true)
                .option("inferSchema", true) // since we need only numeric types
                .csv(path);

        // handling non-numeric fields using vectors
        StringIndexer stringIndexer = new StringIndexer();
        stringIndexer.setInputCol("Gender");
        stringIndexer.setOutputCol("GenderIndex");
        inputData = stringIndexer.fit(inputData).transform(inputData);

        OneHotEncoder oneHotEncoder = new OneHotEncoder();
        oneHotEncoder.setInputCols(new String[]{"GenderIndex"});
        oneHotEncoder.setOutputCols(new String[]{"GenderVector"});
        inputData = oneHotEncoder.fit(inputData).transform(inputData);
        inputData.show();

        // prepare the vector of features
        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[]{"Age", "Height", "Weight", "GenderVector"});
        vectorAssembler.setOutputCol("features");

        inputData = vectorAssembler.transform(inputData)
                .select(col("NoOfReps"), col("features")) // select only required features
                .withColumnsRenamed(Map.of("NoOfReps", "label"));


    }
}
