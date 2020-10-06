import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig


class Analysis():
    def __init__(self,x,y,Fs,FRFlog = False):
        self.x = x
        self.y = y
        self.Fs = Fs
        self.H1 = None
        self.H2 = None
        self.Syy = None
        self.Sxx = None
        self.Sxy = None
        self.FRFlog = FRFlog
        self.welch()
        
    def welch(self,win_size = int(2**12),win = None,overlap =None , Nfft= None):
        x,y = self.x, self.y
        if Nfft== None : Nfft = win_size
        if win == None : win = sig.windows.hann(win_size,sym = False) # bien assymétrique
        if overlap == None : overlap = int(Nfft/2)
        win_size = int(win_size)
        Pw_win =  np.sum(win**2) # calcule de l'énergie de la fenêtre
        N = len(x)
        out = np.zeros(int(win_size))
        pas = win_size-overlap

        Sxx,Syy,Sxy = np.zeros(int(win_size)),np.zeros(int(win_size)),np.zeros(int(win_size))
        for k,i in enumerate(np.arange(0,N-win_size,pas,dtype=int)):
            X   = np.fft.fft(x[i : i+win_size] * win)
            Y   = np.fft.fft(y[i : i+win_size] * win)
            Sxxtmp = np.abs(X)**2/Pw_win
            Sxytmp = np.conj(X)*Y/Pw_win
            Syytmp = np.abs(Y)**2/Pw_win
            Sxx = Sxx*k/(k+1)+Sxxtmp/(k+1)
            Sxy = Sxy*k/(k+1)+Sxytmp/(k+1)
            Syy = Syy*k/(k+1)+Syytmp/(k+1)
            H1 = Sxy/Sxx
            H2 = Syy/np.conj(Sxy)
        self.H1 = H1
        self.H2 = H2
        self.Syy = Syy
        self.Sxx = Sxx
        self.Sxy = Sxy

    def __str__(self):
        fig,ax = plt.subplots(3,2,figsize = (10,10))
        ax1,ax3,ax2 = ax[0,1],ax[1,1],ax[2,1]
        ax4,ax5,ax6 = ax[0,0],ax[1,0],ax[2,0]
        
        
        F = np.arange(len(self.H1))*self.Fs/len(self.H1)
        _t = np.arange(len(self.H1))/self.Fs
        t = np.arange(len(self.x))/self.Fs
        
        ax4.plot(t,self.x,label='x')
        ax4.plot(t,self.y,label='y',alpha= 0.7)
        ax4.set_ylabel(r'$x[n], y[n]$')
        ax4.set_xlabel('time [s]')
        ax4.legend()
        ax4.grid()
        ax4.set_title('waveforms')
        
        ax5.plot(F, 10*np.log10(np.abs(self.Sxx)), label='x')
        ax5.plot(F, 10*np.log10(np.abs(self.Syy)), label='y',alpha= 0.7)
        ax5.set_ylabel(r'$DSP$')
        ax5.set_xlim(0,self.Fs/2)
        ax5.legend()
        ax5.set_title('DSP')
        ax5.grid()
        
        
        ax6.plot(_t,np.real(np.fft.ifft(self.H1)),label='h1')
        ax6.plot(_t,np.real(np.fft.ifft(self.H2)),label='h2',alpha= 0.7)
        ax6.set_xlabel('time [s]')
        ax6.set_ylabel('h[n]')
        ax6.set_title('Impluse response')
        ax6.legend()
        ax6.grid()
        ax1.plot(F, 10*np.log10(np.abs(self.H1)), label='H1')
        ax1.plot(F, 10*np.log10(np.abs(self.H2)), label='H2',alpha= 0.7)
      
        if self.FRFlog : ax1.set_xscale('log')
        ax1.set_ylabel(r'$|H[F]|$ (dB)')
        ax1.set_xlim(0,self.Fs/2)
        ax1.legend()
        ax1.grid()
        ax1.set_title(r'$|H[F]|$')

        ax3.plot(F, np.unwrap(np.angle(self.H1)), label='H1')
        ax3.plot(F, np.unwrap(np.angle(self.H2)), label='H2',alpha= 0.7)
        ax3.set_ylabel(r'arg$(H[F])$')
        ax3.set_xlim(0,self.Fs/2)
        ax3.legend()
        ax3.grid()
        ax3.set_title('Phase')
        ax2.plot(F, np.abs(self.H1/self.H2))
        ax2.set_xlabel('Frequency [Hz]')
        ax1.set_xlabel('Frequency [Hz]')
        ax3.set_xlabel('Frequency [Hz]')
        ax5.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('coherence')
        ax2.set_xlim(0,self.Fs/2)
        ax2.set_title('coherence')
        ax2.grid()
        plt.tight_layout()
        plt.show()
        
        
        return ' '