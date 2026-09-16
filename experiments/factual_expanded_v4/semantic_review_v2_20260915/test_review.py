import copy, unittest
from apply_review import apply_state, amplitude_bounds

class ReviewTests(unittest.TestCase):
    def test_question_binding_and_repeat_consistency(self):
        decisions = {('q1','same'): {'question_id':'q1','answer':'same','label':'matches_reference','pair_id':'a'},
                     ('q2','same'): {'question_id':'q2','answer':'same','label':'contradicts_reference','pair_id':'b'}}
        original = {'label':'unresolved','reference_error':None}; answer={'status':'valid','answer':'same'}
        self.assertEqual(apply_state('q1',answer,original,decisions)['reference_error'],0)
        self.assertEqual(apply_state('q2',answer,original,decisions)['reference_error'],1)
        self.assertEqual(apply_state('q1',answer,original,decisions),apply_state('q1',answer,original,decisions))

    def test_original_and_absence_not_relabelled(self):
        for label,value in [('matches_reference',0),('contradicts_reference',1),('abstention',None),('invalid_output',None)]:
            original={'label':label,'reference_error':value}; before=copy.deepcopy(original)
            result=apply_state('q',{'answer':None},original,{})
            self.assertEqual(result['label'],label); self.assertEqual(result['reference_error'],value)
            self.assertEqual(original,before)

    def test_missing_decision_fails_closed(self):
        with self.assertRaises(KeyError): apply_state('q',{'answer':'x'},{'label':'unresolved','reference_error':None},{})

    def test_stable_semantics_removes_spurious_oscillation(self):
        old=amplitude_bounds([[None,None,None]]*24)
        known=amplitude_bounds([[0,0,0]]*24)
        self.assertEqual(known,[0,0]); self.assertEqual(old,[0,0.5])

if __name__=='__main__': unittest.main()
