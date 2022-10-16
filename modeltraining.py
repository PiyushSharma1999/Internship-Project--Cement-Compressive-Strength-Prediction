from pyexpat import model
from sklearn.model_selection import train_test_split
from Data_Loading import traininig_data_loader
from Data_Preprocessing import preprocessing, clustering
from Find_model import find_model
from File_Operations import file_methods
from App_Logging import logger

class Trainmodel:

    def __init__(self):
        self.write_log = logger.Logging_App()
        self.file_obj = open('Training_Logs/Training_model_logs.txt', 'a+')

    def model_training(self):
        self.write_log.log(self.file_obj,'Training started')

        try:
            # Getting data from source
            data_getter = traininig_data_loader.Data_Getter(self.file_obj,self.write_log)
            data = data_getter.get_data()
         
            # DATA PREPROCESSING 
            preprocessor = preprocessing.Preprocessing(self.file_obj,self.write_log)
            is_null_present = preprocessor.is_null_present(data)
            if(is_null_present):
                data = preprocessor.fill_null_values(data)  # missing values imputation
 
            # Separate label feature
            X,Y = preprocessor.separate_label_feature(data,target_name='concrete_compressive_strength')
 
            X = preprocessor.logTransformation(X)
 
            # Apply clustering
            kmeans = clustering.Clustering(self.file_obj,self.write_log) # initializing object
            number_of_clusters = kmeans.elbow_plot(X) # using the elbow plot to find optimum clusters
 
            # Divide the data into clusters
            X = kmeans.create_cluster(X,number_of_clusters)
 
            # Create a new column in the dataset consisiting of the corresponding cluster assignment
            X['Labels'] = Y
 
            # getting the unique cluster from our datset
            list_of_clusters= X['cluster'].unique()

            for i in list_of_clusters:
                cluster_data = X[X['cluster']==i] # filter the data for one cluster

                # Prepare the feature and Label Column
                cluster_features = cluster_data.drop(['Labels','cluster'],axis=1)
                cluster_label = cluster_data['Labels']

                # Split the data into traininig and test set
                x_train,x_test,y_train,y_test = train_test_split(cluster_features,cluster_label,test_size=1/3,random_state=36)

                x_train_scaled = preprocessor.standardScalar(x_train)
                x_test_scaled = preprocessor.standardScalar(x_test)

                model_finder = find_model.Find_Model(self.file_obj,self.write_log)

                # get best model
                best_model_name , best_model = model_finder.get_best_model(x_train_scaled,y_train,x_test_scaled,y_test)
                file_op = file_methods.File_Operations(self.file_obj,self.write_log)
                save_model = file_op.model_saving(best_model, best_model_name+ str(i))
            self.write_log.log(self.file_obj,'Successful End Of Model Training')

        except Exception:
            self.write_log.log(self.file_obj,'Model training unsuccessful')
            self.file_obj.close()
            raise Exception
