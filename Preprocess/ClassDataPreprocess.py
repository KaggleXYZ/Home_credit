import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import learning_curve
import matplotlib.pyplot as plt
import math

class DataPreprocess:

    def maxQS_process(self,df, columns, quantile_value=0.95):
        print("maxQS_process")
        columns_new = []
        for i in columns:
            if 0 < quantile_value < 1:
                quantile = round(df[i].quantile(quantile_value), 5)
                df[i + str("_q")] = df[i].map(
                    lambda x: x if x < quantile else quantile)
                df[i + str("_q")] = RobustScaler(quantile_range=(5, 95)).fit_transform(df[[i + str("_q")]].fillna(-1))
                columns_new.append(i + str("_q"))
            else:
                quantile = quantile_value
                df[i + str("_q")] = df[i].map(
                    lambda x: x if x < quantile else quantile)
                df[i + str("_q")] = RobustScaler(quantile_range=(5, 95)).fit_transform(df[[i + str("_q")]].fillna(-1))
                columns_new.append(i + str("_q"))
        return df, columns_new

    def robustScaler(self,df, columns, quantile_range=(5, 95)):
        print("robustScaler")
        df[columns] = RobustScaler(quantile_range=quantile_range).fit_transform(df[columns].fillna(-1))
        return df

    def log_process(self,df, columns, bias=1, scaler=False):
        print("log_process")
        columns_new = []
        for i in columns:
            df[i + str("_log")] = df[i].map(lambda x: np.log(x + bias) if x > 0 else 0)
            columns_new.append(i + str("_log"))
            if scaler:
                df[i + str("_log")] = RobustScaler(quantile_range=(5, 95)).fit_transform(df[[i + str("_q")]].fillna(-1))

        return df, columns_new

    def boxcut_process(self,df, columns, q=10, scaler=True):
        print("boxcut_process")
        columns_new = []

        for i in columns:
            df[i + str("_cut")] = pd.qcut(df[i], q, labels=False, duplicates="drop").fillna(-1)
            columns_new.append(i + str("_cut"))
            if scaler:
                df[i + str("_cut")] = RobustScaler(quantile_range=(5, 95)).fit_transform(
                    df[[i + str("_cut")]].fillna(-1))

        return df, columns_new

    def polynomial_process(self,df, columns):
        print("polynomial_process")
        new = ["1"]
        for i in columns:
            new.append(i)
        tmp = PolynomialFeatures(2).fit_transform(df[columns])
        r = tmp.shape[1] - len(columns) - 1
        for i in range(0, r):
            new.append("Polynomial_" + str(i + 1))
        df2 = pd.DataFrame(tmp, columns=new).drop(["1"], axis=1)
        result = pd.concat([df.drop(columns, axis=1), df2], axis=1)
        new_col = df2.columns.values.tolist()
        return result, new_col
    
    def importancePlot(self, dataflag, flag="label", figure=True):
        '''
        输入 dataframe 绘制影响因子排序及图
        :param flag: 标签名称
        :param figure: 是否画图
        :return: 提示建议选取特征数目，仅供参考。
        '''

        label = dataflag[flag]
        data = dataflag.drop(flag, axis=1)
        data1 = np.array(data)
        label = np.array(label).ravel()

        model = ExtraTreesClassifier()
        model.fit(data1, label)

        importance = model.feature_importances_
        std = np.std([importance for tree in model.estimators_], axis=0)
        indices = np.argsort(importance)[::-1]

        featurename = list(data.columns[indices])

        # Print the feature ranking
        print("Feature ranking:")
        importa = pd.DataFrame(
            {'特征权重': list(importance[indices]), '特征名称': featurename})
        print(importa)

        modelnew = SelectFromModel(model, prefit=True)

        print('建议选取的特征数目:', modelnew.transform(data1).shape[1])

        # Plot the feature importances of the forest
        if figure == True:
            plt.figure()
            plt.title("Feature importances")
            plt.bar(range(data1.shape[1]), importance[indices],
                    color="g", yerr=std[indices], align="center")
            plt.xticks(range(data1.shape[1]), indices, rotation=90)
            plt.xlim([-1, data1.shape[1]])
            plt.grid(True)
            plt.show()

    def corrAnaly(self, data):

        corr = data.corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(20, 20))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(110, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, linewidths=.5,
                    cbar_kws={"shrink": .6}, annot=True, annot_kws={"size": 8})
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        sns.plt.show()
        return corr

    def learnCurve(self, modelEstimator, title, data, label, cv=None, train_sizes=np.linspace(0.1, 1.0, 5)):
        '''
        :param estimator: the model/algorithem you choose
        :param title: plot title
        :param x: train data numpy array style
        :param y: target data vector
        :param xlim: axes x lim
        :param ylim: axes y lim
        :param cv:
        :return: the figure
        '''

        train_sizes, train_scores, test_scores = \
            learning_curve(modelEstimator, data, label, cv=cv, train_sizes=train_sizes)

        '''this is the key score function'''
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color='b')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color='g')
        plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label='training score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='cross valid score')
        plt.xlabel('training examples')
        plt.ylabel('score')
        plt.legend(loc='best')
        plt.grid('on')
        plt.title(title)
        plt.show()

    def classifyReport(self, label_tst, pre):
        '''
        交叉验证数据分类报告，0为负样本，1为正样本
        :param label_tst: 标签
        :param pre: 分类结果可能性
        :return:
        '''
        count_tn, count_fp, count_fn, count_tp = 0, 0, 0, 0

        for i in range(len(label_tst)):
            if label_tst[i] == 0:
                if pre[i] < 0.5:
                    count_tn += 1
                else:
                    count_fp += 1
            else:
                if pre[i] < 0.5:
                    count_fn += 1
                else:
                    count_tp += 1

        print('Total:', len(label_tst))
        print('FP被分为好的坏:', count_fp, 'TN正确分类的坏:', count_tn, '坏正确率：',
              round(float(count_tn) / float((count_fp + count_tn)), 3))
        print('FN被分为坏的好:', count_fn, 'TP正确分类的好:', count_tp, '好正确率：',
              round(float(count_tp) / float((count_fn + count_tp)), 3))


    def nanaProcess(self,df):
        """
        利用插值法填充缺失值
        :param df: pandas datframe
        :return: dataframe with processed nan value
        """
        return df.interpolate()

    def get_distances(self, long1, lat1, long2, lat2):
        """
        根据经纬度获得两点之间的距离
        :param long1:
        :param lat1:
        :param long2:
        :param lat2:
        :return: 米
        """
        r = 6378137  # 地球半径

        lat1 = lat1 * math.pi / 180.0
        lat2 = lat2 * math.pi / 180.0

        a = lat1 - lat2
        b = (long1 - long2) * math.pi / 180.0

        sa2 = math.sin(a / 2.0)
        sb2 = math.sin(b / 2.0)
        d = 2 * r * math.asin(math.sqrt(sa2 * sa2 + math.cos(lat1) * math.cos(lat2) * sb2 * sb2))
        return d

    def get_angles(self, long1, lat1, long2, lat2):
        """
        根据经纬度计算两点之间的角度差
        :param long1:
        :param lat1:
        :param long2:
        :param lat2:
        :return:
        """
        y = math.sin(long2 - long1) * math.cos(lat2)

        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(long2 - long1)

        b = math.atan2(y, x)

        b = math.degrees(b)
        if b < 0:
            b = b + 360

        return b

    def read_data(self, address, sep=','):
        """
        PANDAS 分批量读取数据
        :param address:
        :param sep:
        :return:
        """
        reader = pd.read_csv(address, sep=sep, iterator=True, parse_dates=True)
        chunks = []
        loop = True
        i = 1
        while loop:
            try:
                chunk = reader.get_chunk(30000000)
                chunks.append(chunk)
                print(i)
                i += 1
            except StopIteration:
                loop = False
                print('Iteration is stopped.')
        data = pd.concat(chunks, ignore_index=True)
        print("DONE")
        return data

    def time_shift(self, df, sort_a="uid", sort_b="time"):

        '''
        首先根据输入的dataframe 中的 uid 和 time 做排序，这样同一个 UID 对应几个 time，通过计算同一个 UID 下
        对应的各个时间做 shift
        :param df:
        :param sort_a: 输入的标识位
        :param sort_b: 输入的时间变量
        :return: 只保留了最初和最后时间的 uid 和对应的时间。

        examples:
        +---+-----+          +----+----------+
        |uid| time|          |uid |time_shift|
        +---+-----+          +----+----------+
        |123| 2011|      =>  |123 |    1     |
        +---+-----+          +----+----------+
        |123| 2012|
        +---+-----+
        '''
        count = df[sort_a].value_counts()  # 统计 df 中的每一个元素对应的出现频率

        df.sort_values([sort_a, sort_b])  # 对 uid time  排序

        for i in range(count.shape[0]):
            value = count.index[i]
            shift_t = count.values[i] - 1  # 偏移量，一般是依据出现
            df.ix[df[sort_a] == value, "pre"] = df[df[sort_a] == value][sort_b].shift(shift_t)
            df.ix[df[sort_a] == value, "days"] = (df['time'] - df['pre']).dt.days
        # print(df)
        df_new = df.dropna(how="any")  # 这里只保存有值的，本来不是最大一天的都会有空置
        print("DONE")

        return df_new


        
        
        
        

