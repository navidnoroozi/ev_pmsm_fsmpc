
class Inverter:
    def __init__(self, V_dc=400.0):
        self.V_dc = V_dc

    def generateOutputVoltage(self, s_seq):
        """Return inverter pole voltages v_{xN} for each step.
        three-phase: s tuple (sa,sb,sc) per step -> return (va_list, vb_list, vc_list)
        """
        V = self.V_dc/2.0
        va=[]; vb=[]; vc=[]
        for sa,sb,sc in s_seq:
            va.append(V*(2*int(sa)-1))
            vb.append(V*(2*int(sb)-1))
            vc.append(V*(2*int(sc)-1))
        return (va,vb,vc)
