import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class KMeansDriver {
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String inputPath = args[0];
        String outputPath = args[1];
        String centroidsPath = args[2];

        int k = Integer.parseInt(args[3]);
        int maxIterations = 15;

        List<double[]> centroids = loadInitialCentroids(centroidsPath, k);
        for (int i = 0; i < maxIterations; i++) {
            conf.set("centroids", centroidsToString(centroids));

            Job job = Job.getInstance(conf, "KMeans Clustering");
            job.setJarByClass(KMeansDriver.class);
            job.setMapperClass(KMeansMapper.class);
            job.setReducerClass(KMeansReducer.class);
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(Text.class);

            FileInputFormat.addInputPath(job, new Path(inputPath));
            FileOutputFormat.setOutputPath(job, new Path(outputPath + "_" + i));

            long startTime = System.currentTimeMillis();
            if (!job.waitForCompletion(true)) {
                System.exit(1);
            }
            long endTime = System.currentTimeMillis();
            System.out.println("Iteration " + i + " completed in " + (endTime - startTime) + " ms");

            centroids = loadCentroidsFromHDFS(outputPath + "_" + i + "/part-r-00000");
        }
    }

    private static List<double[]> loadInitialCentroids(String path, int k) throws Exception {
        List<double[]> centroids = new ArrayList<>();
        Configuration conf = new Configuration();
        Path centroidsPath = new Path(path);
        BufferedReader br = new BufferedReader(new InputStreamReader(centroidsPath.getFileSystem(conf).open(centroidsPath)));
        String line;
        while ((line = br.readLine()) != null && centroids.size() < k) {
            String[] values = line.split(",");
            double[] centroid = new double[values.length];
            for (int i = 0; i < values.length; i++) {
                centroid[i] = Double.parseDouble(values[i]);
            }
            centroids.add(centroid);
        }
        br.close();
        return centroids;
    }

    private static List<double[]> loadCentroidsFromHDFS(String path) throws Exception {
        List<double[]> centroids = new ArrayList<>();
        Configuration conf = new Configuration();
        Path centroidsPath = new Path(path);
        BufferedReader br = new BufferedReader(new InputStreamReader(centroidsPath.getFileSystem(conf).open(centroidsPath)));
        String line;
        while ((line = br.readLine()) != null) {
            String[] values = line.trim().split("\\s+"); // Using regex to split string, handling spaces
            String[] coordValues = values[1].split(","); // Handling the second part of the coordinates
            double[] centroid = new double[coordValues.length];
            for (int i = 0; i < coordValues.length; i++) {
                centroid[i] = Double.parseDouble(coordValues[i]);
            }
            centroids.add(centroid);
        }
        br.close();
        return centroids;
    }

    private static String centroidsToString(List<double[]> centroids) {
        StringBuilder sb = new StringBuilder();
        for (double[] centroid : centroids) {
            for (double v : centroid) {
                sb.append(v).append(",");
            }
            sb.setLength(sb.length() - 1);
            sb.append(";");
        }
        sb.setLength(sb.length() - 1);
        return sb.toString();
    }
}
