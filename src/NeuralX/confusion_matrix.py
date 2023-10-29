import numpy as np
from numpy.typing import NDArray
from neuralx.neural_network import NeuralNetwork
from math import floor, ceil, sqrt


class ConfusionMatrix:
    def __init__(
        self,
        model: NeuralNetwork,
        data: NDArray,
        labels: NDArray,
        positive_class: int = 1,
        negative_class: int = 0,
        threshold: float = 0.5,
        digits: int = 3
    ) -> None:
        prediction = threshold_function(model.predict(data), threshold)
        labels = np.squeeze(labels)
        self.digits = digits
        self.true_positive = np.sum(labels[labels == positive_class] == prediction[labels == positive_class], dtype=int)
        self.false_negative = np.sum(labels[labels == positive_class] != prediction[labels == positive_class], dtype=int)
        self.false_positive = np.sum(labels[labels == negative_class] != prediction[labels == negative_class], dtype=int)
        self.true_negative = np.sum(labels[labels == negative_class] == prediction[labels == negative_class], dtype=int)
        self.predicted_positive = self.true_positive + self.false_positive
        self.predicted_negative = self.false_negative + self.true_negative
        self.positive = self.true_positive + self.false_negative
        self.negative = self.false_positive + self.true_negative
        self.total = self.positive + self.negative
        self.confusion_matrix = np.array(
            [[self.total, self.predicted_positive, self.predicted_negative],
             [self.positive, self.true_positive, self.false_negative],
             [self.negative, self.false_positive, self.true_negative]]
        )

    def __repr__(self) -> str:
        s000 = "Total = "
        s001 = str(self.total)
        s01 = f"Predicted Positive = {self.predicted_positive}"
        s02 = f"Predicted Negative = {self.predicted_negative}"

        s100 = "Actual Positive = "
        s101 = str(self.positive)
        s11 = str(self.true_positive)
        s12 = str(self.false_negative)

        s200 = "Actual Negative = "
        s201 = str(self.negative)
        s21 = str(self.false_positive)
        s22 = str(self.true_negative)

        s0 = " | ".join([" "*10+s000+s001, s01, s02])
        max_length = len(max(s001, s101, s201, key=len))
        s1 = " | ".join(
            [s100+s101+" "*(max_length-len(s101)),
             " "*floor((len(s01)-len(s11))/2)+s11+" "*ceil((len(s01)-len(s11))/2),
             " "*floor((len(s02)-len(s12))/2)+s12+" "*ceil((len(s02)-len(s12))/2)]
        )
        s2 = " | ".join(
            [s200+s201+" "*(max_length-len(s201)),
             " "*floor((len(s01)-len(s21))/2)+s21+" "*ceil((len(s01)-len(s21))/2),
             " "*floor((len(s02)-len(s22))/2)+s22+" "*ceil((len(s02)-len(s22))/2)]
        )
        return "\n"+("\n"+"-"*len(s0)+"\n").join([s0, s1, s2])+"\n"

    def statistics(self) -> str:
        s1 = "True Positive Rate (Sensitivity)(Recall):"
        s2 = "False Negative Rate:"
        s3 = "False Positive Rate (Fall-out):"
        s4 = "True Negative Rate (Specificity):"
        s5 = "Positive Predictive Value (Precision):"
        s6 = "False Omission Rate:"
        s7 = "False discovery Rate:"
        s8 = "Negative Predictive Value (Precision):"
        s9 = "Accuracy:"
        s10 = "Balanced Accuracy:"
        s11 = "F1 Score:"
        s12 = "Matthews Correlation Coefficient:"
        s13 = "Fowlkes-Mallows Index:"
        s14 = "Jaccard Index:"
        s15 = "Positive Likelihood Ratio:"
        s16 = "Negative Likelihood Ratio:"
        s17 = "Diagnostic Odds Ratio:"
        s18 = "Prevalence:"
        s19 = "Prevalence Threshold:"
        s20 = "Informedness:"
        s21 = "Markedness:"
        globs = globals()
        locs = locals()
        max_length = len(max([eval(f"s{i+1}", globs, locs) for i in range(21)], key=len))
        s1 = ' '*(max_length-len(s1))+s1+' '+str(self.true_positive_rate)
        s2 = ' '*(max_length-len(s2))+s2+' '+str(self.false_negative_rate)
        s3 = ' '*(max_length-len(s3))+s3+' '+str(self.false_positive_rate)
        s4 = ' '*(max_length-len(s4))+s4+' '+str(self.true_negative_rate)
        s5 = ' '*(max_length-len(s5))+s5+' '+str(self.positive_predictive_value)
        s6 = ' '*(max_length-len(s6))+s6+' '+str(self.false_omission_rate)
        s7 = ' '*(max_length-len(s7))+s7+' '+str(self.false_discovery_rate)
        s8 = ' '*(max_length-len(s8))+s8+' '+str(self.negative_predictive_value)
        s9 = ' '*(max_length-len(s9))+s9+' '+str(self.accuracy)
        s10 = ' '*(max_length-len(s10))+s10+' '+str(self.balanced_accuracy)
        s11 = ' '*(max_length-len(s11))+s11+' '+str(self.f1_score)
        s12 = ' '*(max_length-len(s12))+s12+' '+str(self.matthews_correlation_coefficient)
        s13 = ' '*(max_length-len(s13))+s13+' '+str(self.fowlkes_mallows_index)
        s14 = ' '*(max_length-len(s14))+s14+' '+str(self.jaccard_index)
        s15 = ' '*(max_length-len(s15))+s15+' '+str(self.positive_likelihood_ratio)
        s16 = ' '*(max_length-len(s16))+s16+' '+str(self.negative_likelihood_ratio)
        s17 = ' '*(max_length-len(s17))+s17+' '+str(self.diagnostic_odds_ratio)
        s18 = ' '*(max_length-len(s18))+s18+' '+str(self.prevalence)
        s19 = ' '*(max_length-len(s19))+s19+' '+str(self.prevalence_threshold)
        s20 = ' '*(max_length-len(s20))+s20+' '+str(self.informedness)
        s21 = ' '*(max_length-len(s21))+s21+' '+str(self.markedness)
        globs = globals()
        locs = locals()
        return "\n"+"\n".join([eval(f"s{i+1}", globs, locs) for i in range(21)])+"\n"

    @property
    def true_positive_rate(self) -> float:
        return round(self.true_positive/self.positive, self.digits)

    @property
    def sensitivity(self) -> float:
        return self.true_positive_rate

    @property
    def recall(self) -> float:
        return self.true_positive_rate

    @property
    def false_negative_rate(self) -> float:
        return round(self.false_negative/self.positive, self.digits)

    @property
    def false_positive_rate(self) -> float:
        return round(self.false_positive/self.negative, self.digits)

    @property
    def fallout(self) -> float:
        return self.false_positive_rate

    @property
    def true_negative_rate(self) -> float:
        return round(self.true_negative/self.negative, self.digits)

    @property
    def specificity(self) -> float:
        return self.true_negative_rate

    @property
    def positive_predictive_value(self) -> float:
        return round(self.true_positive/self.predicted_positive, self.digits)

    @property
    def precision(self) -> float:
        return self.positive_predictive_value

    @property
    def false_omission_rate(self) -> float:
        return round(self.false_negative/self.predicted_negative, self.digits)

    @property
    def false_discovery_rate(self) -> float:
        return round(self.false_positive/self.predicted_positive, self.digits)

    @property
    def negative_predictive_value(self) -> float:
        return round(self.true_negative/self.predicted_negative, self.digits)

    @property
    def accuracy(self) -> float:
        return round((self.true_positive + self.true_negative)/self.total, self.digits)

    @property
    def informedness(self) -> float:
        return round(self.true_positive_rate + self.true_negative_rate - 1, self.digits)

    @property
    def prevalence(self) -> float:
        return round(self.positive/self.total, self.digits)

    @property
    def prevalence_threshold(self) -> float:
        return round((sqrt(self.recall*self.fallout)-self.fallout)/(self.recall-self.fallout), self.digits)

    @property
    def positive_likelihood_ratio(self) -> float:
        return round(self.true_positive_rate/self.false_positive_rate, self.digits)

    @property
    def negative_likelihood_ratio(self) -> float:
        return round(self.false_negative_rate/self.true_negative_rate, self.digits)

    @property
    def diagnostic_odds_ratio(self) -> float:
        return round(self.positive_likelihood_ratio/self.negative_likelihood_ratio, self.digits)

    @property
    def markedness(self) -> float:
        return round(self.positive_predictive_value + self.negative_predictive_value - 1, self.digits)

    @property
    def balanced_accuracy(self) -> float:
        return round((self.true_positive_rate + self.true_negative_rate)/2, self.digits)

    @property
    def f1_score(self) -> float:
        return round(2*self.true_positive/(self.positive+self.predicted_positive), self.digits)

    @property
    def fowlkes_mallows_index(self) -> float:
        return round(sqrt(self.positive_predictive_value*self.true_positive_rate), self.digits)

    @property
    def matthews_correlation_coefficient(self) -> float:
        return round(
            sqrt(
                self.true_positive_rate*self.true_negative_rate*self.positive_predictive_value*self.negative_predictive_value
                ) -
            sqrt(
                self.false_negative_rate*self.false_positive_rate*self.false_omission_rate*self.false_discovery_rate
                ),
            self.digits)

    @property
    def jaccard_index(self) -> float:
        return round(self.true_positive/(self.true_positive+self.false_negative+self.false_positive), self.digits)


def threshold_function(raw_prediction: NDArray, threshold: float):
    prediction = np.array(raw_prediction, copy=True)
    prediction[raw_prediction < threshold] = 0
    prediction[raw_prediction >= threshold] = 1
    return prediction
