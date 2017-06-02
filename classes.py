class SystemObject:
    def __init__(self, descriptors, decision):
        self.descriptors = descriptors
        self.decision = decision
        self.classifier_decision = None

    def set_classifier_decision(self, decision):
        self.classifier_decision = decision
