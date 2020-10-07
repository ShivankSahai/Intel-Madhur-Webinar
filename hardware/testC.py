
class ch1:
    def __init__(self, f):
        self.x = f

    def ke(self):
        ke1 = self.x

        def ker(w):
            w = w*2
            print(w)

        ker(ke1)


po = ch1(5)
po.ke()
